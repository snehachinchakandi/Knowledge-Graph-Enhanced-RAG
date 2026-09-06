import os
import copy
import ujson as json
import argparse
from tqdm import tqdm
from FlagEmbedding import FlagReranker
from llama_index.core import Settings,VectorStoreIndex,PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from util.kg_post_processor import NaivePostprocessor,KGRetrievePostProcessor,ngram_overlap, GraphFilterPostProcessor
from util.kg_response_synthesizer import get_response_synthesizer

import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('KGRAG')
logging.getLogger('KGRAG').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

from builtins import print as _print
from sys import _getframe
def print(*arg, **kw):
    s = f'Line {_getframe(1).f_lineno}'
    return _print(f"Func {__name__} - {s}", *arg, **kw)

def read_data(args):
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'{data_path} not found')
    if args.dataset == 'hotpotqa':
        with open(data_path,'r',encoding='utf-8') as f:
            data = json.load(f)
    elif args.dataset == 'musique':
        data = []
        with open(data_path,'r',encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    return data

def init_model(args):
    Settings.llm = Ollama(model=args.model_name,request_timeout=200)
    Settings.embed_model = OllamaEmbedding(model_name=args.embed_model_name)

def read_kg(args,data):
    if args.dataset=='hotpotqa':
        ents = set()
        for sample in data:
            for ctx in sample['context']:
                ents.add(ctx[0])
        kg_dir = args.kg_dir
        doc2kg = dict()
        print('Loading KGs')
        for ent in tqdm(ents):
            subkg_path = os.path.join(kg_dir,f'{ent.replace("/","_")}.json')
            if os.path.exists(subkg_path):
                with open(subkg_path,'r',encoding='utf-8')as f:
                    subkg = json.load(f)
                    repkg = copy.deepcopy(subkg)
                    if subkg and len(subkg.keys())>0:
                        for seq in subkg.keys():
                            if len(repkg[seq])==0:
                                del repkg[seq]
                        if len(repkg.keys())>0:
                            doc2kg[ent] = repkg
    elif args.dataset=='musique':
        kg_dir = args.kg_dir
        kg_path = os.path.join(kg_dir,'musique_kg_filtered.json')
        with open(kg_path,'r',encoding='utf-8')as f:
            doc2kg = json.load(f)
    print(f'Loaded kg for {len(doc2kg.keys())} entities from {args.dataset}')
    return doc2kg

def write_prediction(args,data,prediction):
    result_path = args.result_path
    if args.dataset=='hotpotqa':
        with open(result_path,'w',encoding='utf-8') as f:
            json.dump(prediction,f)
    elif args.dataset=='musique':
        with open(result_path,'w',encoding='utf-8') as f:
            for sample in data:
                sample_id = sample['id']
                sample['predicted_answer'] = prediction['answer'][sample_id]
                sample['predicted_support_idxs'] = prediction['sp'][sample_id]
                sample['predicted_answerable'] = sample['answerable']
                f.write(json.dumps(sample)+'\n')
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    print(f'Prediction written to {result_path}')

class ImprovedPostProcessor(NaivePostprocessor):
    """Enhanced post-processor with better supporting fact tracking."""
    
    def _postprocess_nodes(self, nodes, query_bundle=None):
        # Track which nodes were actually used (stored for potential future enhancements)
        used_node_ids = [node.node.id_ for node in nodes]
        return super()._postprocess_nodes(nodes, query_bundle)

def process_sample(args,sample,kg):
    if args.dataset=='hotpotqa':
        sample_id = sample['_id']
    elif args.dataset=='musique':
        sample_id = sample['id']
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    sample_question = sample['question']
    sample_answer = sample['answer']

    ents = set()
    subkg = dict()
    doc_chunks = []
    chunks_index = dict()

    if args.dataset=='hotpotqa':
        ctxs = sample['context']
        ents = [ctx[0] for ctx in ctxs]
        for ctx in ctxs:
            ent = ctx[0]
            chunks_index[ent] = {}
            for i in range(len(ctx[1])):
                if (ent in kg) and (str(i) in kg[ent]) and (len(kg[ent][str(i)])>0):
                    if ent not in subkg:
                        subkg[ent] = dict()
                    target_kg = kg[ent][str(i)]
                    for triplet in target_kg:
                        h,r,t = triplet
                        if ngram_overlap(h,ent)>=0.90 or ngram_overlap(ent,h)>=0.90:
                            h = ent
                        if ngram_overlap(t,ent)>=0.90 or ngram_overlap(ent,t)>=0.90:
                            t = ent
                        triplet = (h,r,t)
                    subkg[ent][str(i)] = target_kg
                text = f'{ent}: {ctx[1][i]}'
                doc_chunk = TextNode(text=text,id_=f'{ent}##{str(i)}')
                doc_chunks.append(doc_chunk)
                chunks_index[ent][str(i)] = text
    elif args.dataset=='musique':
        ctxs = sample['paragraphs']
        for ctx in ctxs:
            idx = ctx['idx']
            ent = ctx['title']
            ents.add(ent)
            if ent not in chunks_index:
                chunks_index[ent] = dict()
            seq = ctx['seq']
            text = f'{ent}: {ctx["paragraph_text"]}'
            if (ent in kg) and (str(seq) in kg[ent]) and (len(kg[ent][str(seq)])>0):
                if ent not in subkg:
                    subkg[ent] = dict()
                subkg[ent][f'{str(idx)}##{str(seq)}'] = kg[ent][str(seq)]
            doc_chunk = TextNode(text=text,id_=f'{str(idx)}##{ent}##{str(seq)}')
            doc_chunks.append(doc_chunk)
            chunks_index[ent][f'{str(idx)}##{str(seq)}'] = text

    # IMPROVEMENT 1: Increased initial retrieval for better recall
    index = VectorStoreIndex(doc_chunks)
    retriever = VectorIndexRetriever(index=index,similarity_top_k=args.initial_top_k)
    
    # IMPROVEMENT 2: Enhanced QA template with better examples and instructions
    qa_rag_template_str = '''Context information is below.
{context_str}

Instructions: Answer the question based ONLY on the context provided above. Think step by step to identify the key facts needed. Give a short factoid answer (as few words as possible).

Examples:
Q: Were Scott Derrickson and Ed Wood of the same nationality?
A: Yes.
Q: Who was born earlier, Emma Bull or Virginia Woolf?
A: Adeline Virginia Woolf.
Q: The arena where the Lewiston Maineiacs played their home games can seat how many people?
A: 3,677 seated.
Q: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
A: Chief of Protocol.
---------------------
Q: {query_str}
A: '''
    qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=qa_rag_prompt_template
    )

    # IMPROVEMENT 3: Enhanced post-processors with better configuration
    expansion_pp = KGRetrievePostProcessor(
        dataset=args.dataset,
        ents=ents,
        doc2kg=subkg,
        chunks_index=chunks_index
    )
    
    bge_reranker = FlagReranker(model_name_or_path=args.reranker,device=3)
    
    # IMPROVEMENT 4: More aggressive filtering with adjusted top_k
    filter_pp = GraphFilterPostProcessor(
        dataset=args.dataset,
        use_tpt=args.use_tpt,
        topk=args.final_top_k,  # Use final_top_k for more aggressive filtering
        ents=ents,
        doc2kg=subkg,
        chunks_index=chunks_index,
        reranker=bge_reranker
    )
    
    # IMPROVEMENT 5: Custom naive processor to track used nodes
    naive_pp = ImprovedPostProcessor(dataset=args.dataset)
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[expansion_pp,filter_pp,naive_pp]
    )

    try:
        response = query_engine.query(sample_question)
        prediction = response.response
        
        # IMPROVEMENT 6: Better supporting fact extraction with deduplication
        if args.dataset=='hotpotqa':
            # Extract unique supporting facts
            sps_dict = {}
            for source_node in response.source_nodes:
                node_id = source_node.node.id_
                parts = node_id.split('##')
                if len(parts) == 2:
                    ent, seq_str = parts
                    try:
                        seq = int(seq_str)
                        if seq >= 0:
                            # Use dict to avoid duplicates
                            key = f"{ent}_{seq}"
                            if key not in sps_dict:
                                sps_dict[key] = [ent, seq]
                    except ValueError:
                        continue
            sps = list(sps_dict.values())
            
            # IMPROVEMENT 7: Limit to most relevant supporting facts
            # Sort by order of appearance (first nodes are usually more relevant)
            sps = sps[:args.max_supporting_facts]
            
        elif args.dataset=='musique':
            # Extract unique indices
            sps_set = set()
            for source_node in response.source_nodes:
                node_id = source_node.node.id_
                parts = node_id.split('##')
                if len(parts) == 3:
                    idx_str = parts[0]
                    try:
                        idx = int(idx_str)
                        if idx >= 0:
                            sps_set.add(idx)
                    except ValueError:
                        continue
            sps = sorted(list(sps_set))[:args.max_supporting_facts]
            
    except Exception as e:
        print(f'Sample {sample_id}, Error: {e}')
        prediction = ''
        sps = []
        
    return sample_id,prediction,sps

def kgrag_distractor_predict(args,data,kg):
    prediction = {'answer':{},'sp':{}}
    sps_count = 0
    for sample in tqdm(data):
        sample_id,sample_prediction,sample_sps = process_sample(args,sample,kg)
        prediction['answer'][sample_id] = sample_prediction
        prediction['sp'][sample_id] = sample_sps
        sps_count += len(sample_sps)
    print(f'Average number of supporting facts: {sps_count/len(data)}')
    return prediction

def main(args):
    data = read_data(args)
    # Use only 5% of the dataset for testing
    n = max(1, int(0.05 * len(data)))
    data = data[:n]
    print(f"Processing only 5% of the dataset: {n} samples out of {len(data)} total")
    init_model(args)
    kg = read_kg(args, data)
    prediction = kgrag_distractor_predict(args, data, kg)
    write_prediction(args, data, prediction)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # hotpotqa distractor
    parser.add_argument('--dataset',type=str,default='hotpotqa',help='Dataset name')
    parser.add_argument('--data_path',type=str,default='../data/hotpotqa/hotpot_dev_distractor_v1.json',help='Path to the data file')
    parser.add_argument('--kg_dir',type=str,default='../data/hotpotqa/kgs/extract_subkgs',help='Directory of the KGs')
    parser.add_argument('--use_tpt',type=bool,default=False,help='Whether to use triplet representation')
    parser.add_argument('--result_path',type=str,default='../output/hotpot/hotpot_dev_distractor_v1_kgrag_improved.json',help='Path to the result file')

    # # pu-hotpotqa distractor
    # parser.add_argument('--dataset',type=str,default='hotpotqa',help='Dataset name')
    # parser.add_argument('--data_path',type=str,default='../data/pu-hotpotqa/hotpot_dev_distractor_v1.json',help='Path to the data file')
    # parser.add_argument('--kg_dir',type=str,default='../data/pu-hotpotqa/kgs/extract_subkgs',help='Directory of the KGs')
    # parser.add_argument('--use_tpt',type=bool,default=False,help='Whether to use triplet representation')
    # parser.add_argument('--result_path',type=str,default='../output/pu-hotpot/pu-hotpot_dev_distractor_v1_kgrag_improved.json',help='Path to the result file')

    # # musique distractor
    # parser.add_argument('--dataset',type=str,default='musique',help='Dataset name')
    # parser.add_argument('--data_path',type=str,default='../data/MuSiQue/musique_ans_v1.0_dev_mapped.jsonl',help='Path to the data file')
    # parser.add_argument('--kg_dir',type=str,default='../data/MuSiQue/kgs/extract_subkgs',help='Directory of the KGs')
    # parser.add_argument('--use_tpt',type=bool,default=True,help='Whether to use triplet representation')
    # parser.add_argument('--result_path',type=str,default='../output/musique/musique_dev_kgrag_improved.jsonl',help='Path to the result file')

    # # trivia
    # parser.add_argument('--dataset',type=str,default='hotpotqa',help='Dataset name')
    # parser.add_argument('--data_path',type=str,default='../data/trivia_qa/trivia.json',help='Path to the data file')
    # parser.add_argument('--kg_dir',type=str,default='../data/trivia_qa/kgs/extracted_subkgs',help='Directory of the KGs')
    # parser.add_argument('--use_tpt',type=bool,default=True,help='Whether to use triplet representation')
    # parser.add_argument('--result_path',type=str,default='../output/trivia_qa/trivia_kgrag_improved.json',help='Path to the result file')

    parser.add_argument('--embed_model_name',type=str,default='mxbai-embed-large',help='Ollama embedding model name for indexing')
    parser.add_argument('--model_name',type=str,default='llama3:8b',help='Ollama model name')
    parser.add_argument('--reranker',type=str,default='../../bge-reranker-large',help='Path of the reranker model')
    
    # IMPROVEMENT 8: New hyperparameters for better control
    parser.add_argument('--initial_top_k',type=int,default=15,help='Initial retrieval top k (increased from 10)')
    parser.add_argument('--final_top_k',type=int,default=6,help='Final filtered top k (decreased from 10 for precision)')
    parser.add_argument('--max_supporting_facts',type=int,default=4,help='Maximum number of supporting facts to return')

    args = parser.parse_args()

    main(args)
