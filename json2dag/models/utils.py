
from typing import Callable, Dict
import pymc as pm
import numpy as np


def process_docs(fn: Callable)-> Dict[str, str|int]:
  """Process the docstring of a function to extract the operation name and number of arguments.

  Args:
      fn (Callable): 
        function with appropriate docstring format
        ''' 
        Name of the function
        -----
        n_args = 1
        -----
        ...
        '''
  Returns:
      _type_: _description_
  """
  docs = fn.__doc__.split("\n")
  docs = [doc.strip() for doc in docs if doc.strip().replace("-", "") != ""]
  name = fn.__name__
  n_args = int(docs[1].split("=")[1].strip())
  
  return {'op_name': name, 'op_n_args': n_args}


def partial_with_docs(func, **kwargs):
  def partial_func(*args):
    return func(*args, **kwargs)
  partial_func.__doc__ = func.__doc__
  partial_func.__name__ = f"{func.__name__} {';'.join([f'{k}={v}' for k, v in kwargs.items()])}"
  return partial_func

def pipe(*fns):
  
  def _pipe(data, *args, **kwargs):
    args_left = 0
    for fn in fns:
      n_args = process_docs(fn)['op_n_args']
      data = fn(data, *args[args_left:args_left+n_args])
      args_left += n_args
    return data
  n_args = sum(process_docs(fn)['op_n_args'] for fn in fns)
  _pipe.__name__ = f"{' -> '.join([fn.__name__ for fn in fns])}"
  _pipe.__doc__ = f"""
  Data Pipeline
  ------
  n_args = {n_args}
  """
  return _pipe

VALID_PRIOR_CONSTRAINTS = {
  'positive': partial_with_docs(pm.Truncated, dist=pm.Normal.dist(mu=0., sigma=3.), lower=0, upper=np.Inf),
  'negative': partial_with_docs(pm.Truncated, dist=pm.Normal.dist(mu=0., sigma=3.), lower=-np.Inf, upper=0),
  'zero_one': partial_with_docs(pm.Beta, alpha=1, beta=1)
}

def make_appropriate_prior(name: str, n_args: int, constraint: str|list[str]|None, offset:int=0):
  if constraint is None:
    return [pm.Normal(f"{name}_arg_{i+offset}", mu=0, sigma=1) for i in range(n_args)]
  if isinstance(constraint, str):
    return [VALID_PRIOR_CONSTRAINTS[constraint](f"{name}_arg_{i+offset}") for i in range(n_args)]
  if isinstance(constraint, list):
    assert len(constraint) == n_args
    return [make_appropriate_prior(name, 1, c, offset=i)[0] for i, c in enumerate(constraint)]
  