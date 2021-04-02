echo "Making distance matrix."
python3 make_distance_matrix.py
echo "Getting ID's of 8 closest neighbors to each municipality."
python3 get_min_ids.py
echo "Saving arrays with target municipality and their neighbors' data for input into scoailSigNet."
python3 save_VB_arrays.py
echo "Training value-based socialSigNet."
python3 socialSigN_VB_training.py
echo "Evaluating value-based socialSigNet."
python3 socialSigN_VB_eval.py