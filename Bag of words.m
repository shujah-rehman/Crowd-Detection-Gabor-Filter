setDir  = fullfile('dataset');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainingSet,testSet] = splitEachLabel(imds,0.3,'randomize');
bag = bagOfFeatures(imds);
classifier = trainImageCategoryClassifier(imds,bag);
img = imread(fullfile(setDir,'crowd','1.jpg'));
[labelIdx, score] = predict(classifier,img);
