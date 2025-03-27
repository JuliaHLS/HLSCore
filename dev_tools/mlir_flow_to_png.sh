#!/bin/bash

input_file="$1"
output_file="$2"

echo "Reading $input_file and converting to dot graph, and then dot graph to $output_file"

mlir-opt --view-op-graph $input_file 2> temp.dot
dot -Tpng temp.dot -o $output_file

echo "Success"
