#!/bin/bash

# Monolingual task
python preprocess_CoSimLex.py --input path_to_input --output path_to_output
python preprocess_SCWS.py --input path_to_input --output path_to_output
python preprocess_USim.py --input_xml path_to_input_xml_file --input_annotation path_to_input_annotation_file --output path_to_output
