import numpy as np
import pydantic as pyd
from typing import Callable, Optional
import graphviz as gv
import pymc as pm
import pytensor.tensor as pt


from json2dag.models.relations import EdOp, Linear
from json2dag.models.utils import process_docs, make_appropriate_prior

class AbstractNode(pyd.BaseModel):
    pass

class Node(AbstractNode):
    name: str
    value: float|list
    parents: set[AbstractNode] = []
    children: set[AbstractNode] = []
    observed: bool = False
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def __hash__(self):
        return hash(self.name)
    
    def __str__(self):
        return f'{self.name} = {self.value}'

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return self.value + other.value

    def __sub__(self, other):
        return self.value - other.value

    def __mul__(self, other):
        return self.value * other.value

    def __truediv__(self, other):
        return self.value / other.value

    def __pow__(self, other):
        return self.value ** other.value

class Edge(pyd.BaseModel):
    
    parent: Node
    child: Node
    op: Callable= Linear()
    prior_constraint: Optional[str|list[str|None]] = None

    def __init__(self, **data):
        super().__init__(**data)

    def __str__(self):
        return f'{self.parent.name} -> {self.child.name}'

    def __repr__(self):
        return str(self)
      
    @pyd.computed_field
    @property
    def op_n_args(self)->int:
        return self.op.n_args

    @pyd.computed_field
    @property
    def op_name(self)->str:
        return self.op.op_name
    
    def apply(self, *args):
        try:
            assert len(args) == self.op_n_args
        except AssertionError:
            raise ValueError(f'Expected {self.op_n_args} arguments, got {len(args)}')
        
        return self.op(self.parent.value, *args)
    
    def assign(self):
        self.child.value += self.apply()
        
    def __hash__(self):
        return hash(f"{self.parent} {self.child}")
      
class DAG:

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.model = None
        self.trace = None
    
    def display(self):
        dg = gv.Digraph()
        for node in self.nodes:
            dg.node(node.name, f"{node.name}={node.value}")
        for edge in self.edges:
            dg.edge(edge.parent.name, edge.child.name, label=edge.op_name)
        return dg
  
    def __repr__(self):
        return self.display().source
  
    @pyd.computed_field
    @property
    def graph(self)->bytes:
        return self.display().pipe("png")

    @staticmethod
    def dag_from_causes_dict(causes: dict) -> 'DAG':

        def _handle_effects(d: str|dict):
            if isinstance(d, str):
                return {"name": d}
            return d
        causes = {k: list(map(_handle_effects, v)) for k, v in causes.items()}
        nodes = {name: Node(name=name, value=0) for name in causes.keys()}

        nodes.update({name["name"]: Node(name=name["name"], value=0) for effects in causes.values() for name in effects})
        edges = set()
        for cause, effects in causes.items():
            for effect in effects:
                effect_ = effect.copy()
      
                edges.add(Edge(parent=nodes[cause], child=nodes[effect_.pop("name")], **effect_))
        #edges = {Edge(parent=nodes[cause], child=nodes[effect.pop("name")], **effect) for cause, effects in causes.items() for effect in effects}
        return DAG(nodes=nodes.values(), edges=edges)

    def model_from_dag(self, observations=None):
        dag = self
        n_obs = len(observations[list(observations.keys())[0]]) if observations is not None else 0
        with pm.Model(coords = {'nodes': [node.name for node in dag.nodes if node.observed]}|{f"{edge}_dims": edge.op.args for edge in dag.edges}, 
                coords_mutable=None if observations is None else {'obs': range(n_obs)}) as model:
    
            if observations is None:
                observations = {node.name: None for node in dag.nodes}
    
            else:
                observations = {node.name: pm.MutableData(f'obs_{node.name}', observations.get(node.name, None), dims='obs') for node in dag.nodes}
            for node in dag.nodes:
                node.value = observations[node.name]

            observation_noise = pm.HalfCauchy('observation_noise',1, dims='nodes')
    
            node_model = {}
            for edge in dag.edges:
                if edge.child.observed:
                    beta = make_appropriate_prior(f"{edge}", edge.op_n_args, edge.prior_constraint)  
                    pm.Deterministic(f"{edge}", pt.stack(beta), dims=f'{edge}_dims')
                    node_model[edge.child.name] = node_model.get(edge.child.name, 0) + edge.apply(*beta)
            
            
            i = 0
            for node in dag.nodes:
                if node.observed:
                    node_mean = pm.Normal(f"mu_{node.name}", mu=0, sigma=10)
                    pm.Normal(f"{node.name}", mu=node_model.get(node.name, 0)+node_mean, sigma=observation_noise[i], observed=observations[node.name], dims='obs')
                    i += 1
            self.model = model
        return model
    
    def sample(self, draws, **kwargs):
        if self.model is None:
            self.model_from_dag()
        with self.model:
            self.trace = pm.sample(draws, **kwargs)
    
    def summary(self, **kwargs):
        if self.trace is None:
            raise ValueError('No trace found. Please sample first')
        if not kwargs:
            return pm.summary(self.trace, var_names=[f"{edge}" for edge in self.edges if edge.child.observed]+[f'mu_{node.name}' for node in self.nodes if node.observed])
        return pm.summary(self.model, **kwargs)

    def get_paths(self, start, end, nodes_checked=None):
        paths = []
        if nodes_checked is None:
            nodes_checked = []
        
        for node in self.nodes:
            
            if node.name in nodes_checked:
                continue
            
            if node.name == start:
                for edge in self.get_node_edges(node.name):
                    if edge.child.name == end or edge.parent.name == end:
                        paths.append([edge])
                    else:
                        nodes_checked.append(node.name)
                        for path in self.get_paths(edge.child.name, end, nodes_checked):
                            paths.append([edge] + path)
                        for path in self.get_paths(edge.parent.name, end, nodes_checked):
                            paths.append([edge] + path)
             
        return paths
    
    def get_node_noncausal_edges(self, node_name):
        edges = []
        for edge in self.edges:
            if (edge.child.name == node_name): 
                edges.append(edge)
        return edges

    def get_node_edges(self, node_name):
        edges = []
        for edge in self.edges:
            if (edge.parent.name == node_name) or (edge.child.name == node_name):
                edges.append(edge)
        return edges
        
    def get_node_causal_edges(self, node_name):
        edges = []
        for edge in self.edges:
            if (edge.parent.name == node_name):
                edges.append(edge)
        return edges
    
    @staticmethod
    def dag_from_edge_list(edge_list):
        causes = {}
        for edge in edge_list:
            if edge.parent.name in causes:
                causes[edge.parent.name].append(edge.child.name)
            else:
                causes[edge.parent.name] = [edge.child.name]
        return DAG.dag_from_causes_dict(causes)
    
