
def compact_fs():
  old_graph = {}
  id_map = {}
  node_num = 9
  path = '../data/test.txt'
  with open(path, 'r') as f:
    head = 0
    id_cnt = 0
    for line in f:
      if head > 0:
        head -= 1
        continue
      id1, id2 = line.split()
      id1 = int(id1)
      id2 = int(id2)
      if id1 not in old_graph.keys():
        old_graph[id1] = []
      old_graph[id1].append(id2)
      if id1 not in id_map.keys():
        id_map[id1] = id_cnt
        id_cnt += 1
      if id2 not in id_map.keys():
        id_map[id2] = id_cnt
        id_cnt += 1
  
  new_graph = {}
  for src in old_graph.keys():
    new_src = id_map[src]
    new_graph[new_src] = []
    for des in old_graph[src]:
      new_graph[new_src].append(str(id_map[des]))
  
  new_path = '../data/test.compact.txt'
  with open(new_path, 'w') as f:
    line = str(node_num) + '\n'
    f.write(line)
    for id in range(node_num):
      if id not in new_graph.keys():
        line = '\n'
      else:
        line = ' '.join(new_graph[id]) + '\n'
      f.write(line)

if __name__ == '__main__':
  compact_fs()