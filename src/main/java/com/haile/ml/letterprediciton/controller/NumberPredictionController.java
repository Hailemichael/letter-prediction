package com.haile.ml.letterprediciton.controller;


import java.io.File;
import java.math.BigDecimal;
import java.util.HashMap;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;

import com.haile.ml.letterprediciton.deeplearning4j.NumberTrainer;
import com.haile.ml.letterprediciton.model.Prediction;

@Controller
@RequestMapping("/letter-prediction")
public class NumberPredictionController {
	private static final Logger logger = LoggerFactory.getLogger(NumberPredictionController.class);
	private static final String basePath = "C:\\Users\\Haile\\Documents\\Haile\\machine-learning\\mnist_png";
	
	private NumberTrainer numberTrainer;
	
	@SuppressWarnings("unchecked")
	@RequestMapping(value="/predict/number", method=RequestMethod.POST, headers = "content-type=multipart/form-data")
    public @ResponseBody Prediction predictLetter(@RequestParam(value="file", required=true) MultipartFile file) throws Exception {
		logger.info("Prediction begins...");
    	if (file.isEmpty()) {
    		return new Prediction(-1, null);
    	}
    	numberTrainer = new NumberTrainer();    	
    	HashMap<String, Object> predictionMap = numberTrainer.predictNumber(file.getInputStream(), basePath + File.separator + "minist-model.zip");
    	
		return new Prediction((Integer) predictionMap.get("predicted"), (HashMap<String, BigDecimal>) predictionMap.get("output"));
    }
	
	@RequestMapping(value="/train/number", method=RequestMethod.GET)
    public @ResponseBody String trainNumber() {
		numberTrainer = new NumberTrainer();
		Runnable myrunnable = new Runnable() {
			public void run() {
				numberTrainer.generateTrainedModel(basePath + File.separator + "training", basePath + File.separator + "minist-model.zip");
			}
		};
		new Thread(myrunnable).start();
		
		return "Training initiated. Trained model will be generated in few minutes!";
    }
}
