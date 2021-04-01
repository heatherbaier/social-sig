echo "Training socialSig No Drop model"
python3 socialSigNoDrop_train.py
echo "Evaluating socialSig No Drop model"
python3 socialSigNoDrop_eval.py
echo "Training socialSig Value-Based Neighbors model"
python3 neighbors/socialSigN_VB_training.py
echo "Evaluating socialSig Value-Based Neighbors model"
python3 neighbors/socialSigN_VB_eval.py
echo "Making model plots"
python3 cnn_model_plots.py
echo "Done"