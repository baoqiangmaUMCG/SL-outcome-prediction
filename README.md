# SL-outcome-prediction
This is the code using self-learning based method to extract CT image features and build cox model for outcome prediction on OPC-Radiomics dataset 

requirements:
pytorch==1.6.0 lifelines


1. Data preparation code

  cd Data_preprocessing
  
  python OriFile2Digit_opcradiomics.py   # clinical and outcome preprocessing
  
  python org_OPCRadiomics_3Dpatch.py   # select patients who have GTV
  
  python Crop3DPatches_ClinialCategory_OPCradiomics.py   # save clinical and crop image data into pickle file 
  
2. Autoencoder training for get image features

  cd Autoencoder
  
  python PTmain_3DFeatureExtr.py --input_type 3
  
3. outcome prediction model

  cd Outcome_prediction
  
  python forward_clinical_feature_selection.py --sub 1 # select clinical predictors
  
  python forward_combined_feature_selection.py --sub 1 # select predictors from clincial model output and image features
  
  python CoxRegresion.py --outcome 0  # build clinical and combined models, and do risk stratification and calibration curves
