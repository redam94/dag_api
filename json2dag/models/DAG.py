import numpy as np
import pydantic as pyd
from typing import Callable, Optional
import graphviz as gv
import pymc as pm

from json2dag.models.relations import linear


class AbstractNode(pyd.BaseModel):
    pass

class Node(AbstractNode):
    name: str
    value: float|list
    parents: set[AbstractNode] = []
    children: set[AbstractNode] = []
    messurement_error: bool = False
    
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

    def __radd__(self, other):
        return other.value + self.value

    def __rsub__(self, other):
        return other.value - self.value

    def __rmul__(self, other):
        return other.value * self.value

    def __rtruediv__(self, other):
        return other.value / self.value

    def __rpow__(self, other):
        return other.value ** self.value

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __abs__(self):
        return abs(self.value)

    def __neg__(self):
        return -self.value

    def __pos__(self):
        return +self.value

    def __invert__(self):
        return ~self.value

    def __round__(self, n):
        return round(self.value, n)

    def __floor__(self):
        return np.floor(self.value)

    def __ceil__(self):
        return np.ceil(self.value)

    def __trunc__(self):
        return
      
class Edge(pyd.BaseModel):
    parent: Node
    child: Node
    op: Callable = linear
    prior_constraint: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)

    def __str__(self):
        return f'{self.parent.name} -> {self.child.name}'

    def __repr__(self):
        return str(self)
      
    def _process_docs(self):
      docs = self.op.__doc__.split("\n")
      docs = [doc.strip() for doc in docs if doc.strip().replace("-", "") != ""]
      name = self.op.__name__
      n_args = int(docs[1].split("=")[1].strip())
      return {'op_name': name, 'op_n_args': n_args}
    
    @pyd.computed_field
    @property
    def op_n_args(self)->int:
        return self._process_docs()['op_n_args']

    @pyd.computed_field
    @property
    def op_name(self)->str:
        return self._process_docs()['op_name']
    
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
      
class DAG(pyd.BaseModel):
  nodes: Optional[set[Node]] 
  edges: Optional[set[Edge]]
  
  def __init__(self, **data):
    super().__init__(**data)
  
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
  
  
def dag_from_causes_dict(causes):
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

def model_from_dag(dag, observations=None):
  
  n_obs = len(observations[list(observations.keys())[0]]) if observations is not None else 0
  with pm.Model(coords = {'nodes': [node.name for node in dag.nodes]}, 
                coords_mutable=None if observations is None else {'obs': range(n_obs)}) as model:
    
    if observations is None:
      observations = {node.name: None for node in dag.nodes}
    
    else:
      observations = {node.name: pm.MutableData(f'obs_{node.name}', observations.get(node.name, None), dims='obs') for node in dag.nodes}
      for node in dag.nodes:
        node.value = observations[node.name]
        
      
    
    observation_noise = pm.HalfNormal('observation_noise', 1, dims='nodes')
    
    node_model = {}
    for edge in dag.edges:
      if edge.prior_constraint is None:
        beta = [pm.Normal(f'{edge}_arg_{i}', mu=0, sigma=1) for i in range(edge.op_n_args)]
    
      if edge.prior_constraint == 'positive':
        beta = [pm.Beta(f'{edge}_arg_{i}', .5, .5) for i in range(edge.op_n_args)]
        
      
      node_model[edge.child.name] = node_model.get(edge.child.name, 0) + edge.apply(*beta)
    
    
    for i, node in enumerate(dag.nodes):
      node_mean = pm.Normal(f"mu_{node.name}", mu=0, sigma=1)
      pm.Normal(f"{node.name}", mu=node_model.get(node.name, 0)+node_mean, sigma=observation_noise[i], observed=observations[node.name], dims='obs')
      
  return model