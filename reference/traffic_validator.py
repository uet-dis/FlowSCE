"""Schema-driven RAW-feature admissibility, not a packet-realizability certificate.

There is no implied mapping to CIC/DoH/UNSW fields. Feature names, units, limits,
and dependent-statistic identities must be supplied from the exact extractor.
Neither clipping nor silent imputation is performed.
"""
from __future__ import annotations
import math
from typing import Mapping


def validate_raw(candidate: Mapping[str,float], original: Mapping[str,float], schema:dict) -> list[str]:
    fields=schema['fields']; errors=[]
    if set(candidate)!=set(fields) or set(original)!=set(fields):
        raise ValueError('Candidate, original and schema must have exactly the same complete field set.')
    for name,spec in fields.items():
        if not spec.get('unit') or 'mutable' not in spec:raise ValueError('Missing unit or editability for '+name)
        x=float(candidate[name]); x0=float(original[name])
        if not math.isfinite(x) or not math.isfinite(x0):errors.append(name+': nonfinite');continue
        if not spec['mutable'] and x!=x0:errors.append(name+': immutable')
        if spec.get('lower') is not None and x<float(spec['lower']):errors.append(name+': below bound')
        if spec.get('upper') is not None and x>float(spec['upper']):errors.append(name+': above bound')
        if spec.get('integer',False) and abs(x-round(x))>float(spec.get('integer_atol',1e-8)):errors.append(name+': noninteger')
    for rel in schema.get('relations',[]):
        names=[rel[k] for k in (['rate','total','duration'] if rel['kind']=='rate' else ['left','right'])]
        if any(k not in fields for k in names):raise ValueError('Relation refers to an undeclared field.')
        if rel['kind']=='rate':
            rate,total,duration=[float(candidate[n]) for n in names]
            if duration<=0:
                errors.append('rate relation: nonpositive duration requires an explicitly different extractor convention');continue
            expected=float(rel['factor'])*total/duration
            if not math.isclose(rate,expected,rel_tol=float(rel.get('rtol',1e-6)),abs_tol=float(rel.get('atol',1e-8))):errors.append('rate relation: '+names[0])
        elif rel['kind']=='leq':
            if float(candidate[names[0]])>float(candidate[names[1]]):errors.append('ordering: '+names[0])
        else:raise ValueError('Unsupported relation kind: '+str(rel['kind']))
    return errors
