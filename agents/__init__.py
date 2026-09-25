from agents.tql import TQLAgent
from agents.mtql import MTQLAgent
from agents.tql_visual_history import TQLVisualHistoryAgent
from agents.tql_ablation import TQLAblationAgent
from agents.q_transformer import QTransformerAgent
from agents.pac_fql_learner import PACFQLActorAgent
from agents.bc_transformer import BCTransformerAgent
from agents.bc_flow_transformer import BCFlowTransformerAgent
from agents.new_bc_flow_transformer import NewBCFlowTransformerAgent
from agents.mtql_transformer import MTQLTransformerAgent
from agents.mtql_transformer_real import MTQLTransformerRealAgent
from agents.mtql_mlp import MTQLMLPAgent
from agents.mtql_mlp_real import MTQLMLPRealAgent
from agents.new_bc_flow_transformer_real import NewBCFlowTransformerRealAgent
from agents.mtql_transformer_language_real import MTQLTransformerLanguageRealAgent
from agents.mtql_mlp_language_real import MTQLMLPLanguageRealAgent
from agents.new_bc_flow_transformer_language_real import NewBCFlowTransformerLanguageRealAgent
from agents.mtql_transformer_v2 import MTQLTransformerV2Agent


agents = dict(
    tql=TQLAgent,
    mtql=MTQLAgent,
    tql_visual_history=TQLVisualHistoryAgent,
    tql_ablation=TQLAblationAgent,
    q_transformer=QTransformerAgent,
    pac_fql_learner=PACFQLActorAgent,
    bc_transformer=BCTransformerAgent,
    bc_flow_transformer=BCFlowTransformerAgent,
    new_bc_flow_transformer=NewBCFlowTransformerAgent,
    mtql_transformer=MTQLTransformerAgent,
    mtql_transformer_real=MTQLTransformerRealAgent,
    mtql_mlp=MTQLMLPAgent,
    mtql_mlp_real=MTQLMLPRealAgent,
    new_bc_flow_transformer_real=NewBCFlowTransformerRealAgent,
    mtql_transformer_language_real=MTQLTransformerLanguageRealAgent,
    mtql_mlp_language_real=MTQLMLPLanguageRealAgent,
    new_bc_flow_transformer_language_real=NewBCFlowTransformerLanguageRealAgent,
    mtql_transformer_v2=MTQLTransformerV2Agent,
)
