from collections import namedtuple
import pickle

obj1 = namedtuple('Object', ('cls_id', 'track_id', 'bbox'))(10, 1, ['xmin', 'ymin', 'xmax', 'ymax'])


result_str = []
for i in range(3):
    result_str.append(obj1)

print(result_str)

d = pickle.dumps(obj1._asdict())
print(d)

