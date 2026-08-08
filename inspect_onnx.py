import onnx

model = onnx.load("/home/goku/hamza-workdir/Harsh-Feature-Bench/aliked-n16rot-16k.onnx")

# Build a lookup: output_name -> node
output_to_node = {}
for n in model.graph.node:
    for o in n.output:
        output_to_node[o] = n

# Trace backwards from ScatterND_3 looking for MaxPool
def trace_back(output_name, depth=0, visited=None):
    if visited is None:
        visited = set()
    if output_name in visited or depth > 15:
        return
    visited.add(output_name)
    
    if output_name in output_to_node:
        n = output_to_node[output_name]
        prefix = "  " * depth
        print(f"{prefix}{n.op_type} ({n.name})")
        if n.op_type == "MaxPool":
            print(f"{prefix}  *** THIS IS LIKELY THE NMS MAXPOOL ***")
        for inp in n.input:
            trace_back(inp, depth + 1, visited)

print("Tracing from ScatterND_3:")
trace_back("/ScatterND_3_output_0")