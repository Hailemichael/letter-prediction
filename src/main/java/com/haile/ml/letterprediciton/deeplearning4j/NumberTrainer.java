package com.haile.ml.letterprediciton.deeplearning4j;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import javax.imageio.ImageIO;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NumberTrainer {
	
	private static final Logger logger = LoggerFactory.getLogger(NumberTrainer.class);
	
	private MultiLayerNetwork model;
	
	int height = 28;
	int width = 28;
	int channels = 1; // single channel for grayscale images
	int outputNum = 10; // 10 digits classification
	int batchSize = 54;
	int nEpochs = 1;
	int iterations = 1;
	int seed = 1234;
    Random randNumGen = new Random(seed);
	
	public NumberTrainer() {
	}
	
	public void generateTrainedModel(String dataSource, String modelPath) {
		// vectorization of train data
	    File trainData = new File(dataSource);
	    FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
	    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
	    ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
	    try {
			trainRR.initialize(trainSplit);
		} catch (IOException e) {
			logger.error("Exception while initializing number image data for training");
		}
	    DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

	    // pixel values scaling from 0-255 to 0-1
	    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
	    scaler.fit(trainIter);
	    trainIter.setPreProcessor(scaler);

	    logger.info("Network configuration and training...");
	    Map<Integer, Double> lrSchedule = new HashMap<Integer, Double>();
	    lrSchedule.put(0, 0.06); // iteration #, learning rate
	    lrSchedule.put(300, 0.05);
	    lrSchedule.put(500, 0.029);
	    lrSchedule.put(700, 0.0043);
	    lrSchedule.put(1000, 0.001);

	    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	        .seed(seed)
	        .iterations(iterations)
	        .regularization(true).l2(0.0005)
	        .learningRate(.01)
	        .learningRateDecayPolicy(LearningRatePolicy.Schedule)
	        .learningRateSchedule(lrSchedule) // overrides the rate set in learningRate
	        .weightInit(WeightInit.XAVIER)
	        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	        .updater(Updater.NESTEROVS)
	        .list()
	        .layer(0, new ConvolutionLayer.Builder(5, 5)
	            .nIn(channels)
	            .stride(1, 1)
	            .nOut(20)
	            .activation(Activation.IDENTITY)
	            .build())
	        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
	            .kernelSize(2, 2)
	            .stride(2, 2)
	            .build())
	        .layer(2, new ConvolutionLayer.Builder(5, 5)
	            .stride(1, 1) // nIn need not specified in later layers
	            .nOut(50)
	            .activation(Activation.IDENTITY)
	            .build())
	        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
	            .kernelSize(2, 2)
	            .stride(2, 2)
	            .build())
	        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
	            .nOut(500).build())
	        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
	            .nOut(outputNum)
	            .activation(Activation.SOFTMAX)
	            .build())
	        .setInputType(InputType.convolutionalFlat(28, 28, 1)) // InputType.convolutional for normal image
	        .backprop(true).pretrain(false).build();

	    model = new MultiLayerNetwork(conf);
	    model.init();
	    model.setListeners(new ScoreIterationListener(10));

	    logger.debug("Total num of params: {}", model.numParams());
	    for (int i = 0; i < model.getnLayers(); i++) {
	    	logger.debug("Layer {}, num of params: {}", i, model.getLayer(i).numParams());
	    }

	    // evaluation while training (the error/score should go down)
	    for (int i = 0; i < nEpochs; i++) {
	    	model.fit(trainIter);
	      logger.info("Completed epoch {}", i);
	      trainIter.reset();
	    }

	    try {
			ModelSerializer.writeModel(model, new File(modelPath), true); 
		} catch (IOException e) {
			logger.error("Exception while writing model to file.");
		}	    
	}
	
	public HashMap<String, Object> predictNumber (InputStream is, String modelPath) throws Exception {
		Image tmp = null;
		try {
			tmp = ImageIO.read(is);
		} catch (IOException e) {
			throw new Exception("Unable to read image file.");
		}
	    
	    BufferedImage scaledImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
	    Graphics graphics = scaledImg.getGraphics();
	    graphics.drawImage(tmp, 0, 0, null);
	    graphics.dispose();
	    
	    NativeImageLoader loader = new NativeImageLoader(28, 28, 1, true);
	    INDArray image = null;
		try {
			image = loader.asRowVector(scaledImg);
		} catch (IOException e) {
			throw new Exception("Unable to load image for prediction.");
		}
	    ImagePreProcessingScaler testScaler = new ImagePreProcessingScaler(0, 1);
	    testScaler.transform(image);
	    
	    model = loadTrainedModel(modelPath);
	    INDArray output = model.output(image);
	    int[] prediction = model.predict(image);
	    
	    logger.info("Prediction: {}\noutput: {}\n", prediction[0] , output);
	    HashMap<String, Object> predictionMap = new HashMap<String, Object> ();
	    predictionMap.put("predicted", prediction[0]);
	    
	    double[] outputArray = output.data().asDouble();
	    HashMap<String, BigDecimal> outputMap = new HashMap<String, BigDecimal>();
	    for (int i = 0; i < outputArray.length; i++) {
	    	outputMap.put(String.valueOf(i), new BigDecimal(outputArray[i]*100).setScale(2, RoundingMode.HALF_UP));
	    } 
	    
	    predictionMap.put("output", outputMap);
		return predictionMap;
	}
	
	private MultiLayerNetwork loadTrainedModel(String path) throws Exception {
		File modelFile = new File(path);
	    if (!modelFile.exists()) {
	    	logger.error("Trained model not found at specified path: " + path);
	    	throw new Exception("Trained model not found at specified path: " + path);
	    }
			return ModelSerializer.restoreMultiLayerNetwork(modelFile);
	}

}
