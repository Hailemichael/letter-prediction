package com.haile.ml.letterprediciton.controller;


import java.io.File;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;

import com.haile.ml.letterprediciton.deeplearning4j.LetterTrainer;
import com.haile.ml.letterprediciton.model.Prediction;

@Controller
@RequestMapping("/letter-prediction")
public class LetterPredictionController {
	private static final Logger logger = LoggerFactory.getLogger(LetterPredictionController.class);
	private static final String basePath = "C:\\Users\\Haile\\Documents\\Haile\\machine-learning\\mnist_png";
	
	private LetterTrainer letterTrainer;
	
	@RequestMapping(value="/predict/letter", method=RequestMethod.POST, headers = "content-type=multipart/form-data")
    public @ResponseBody Prediction predictLetter(@RequestParam(value="file", required=true) MultipartFile file) throws Exception {
		logger.info("Prediction begins...");
    	if (file.isEmpty()) {
    		return new Prediction(-1, null);
    	}
    	letterTrainer = new LetterTrainer();    	
    	Integer predicted = letterTrainer.predictNumber(file.getInputStream(), basePath + File.separator + "minist-model.zip");//Change integer to char
    
		return new Prediction(predicted, null);
    }
	
	@RequestMapping(value="/train/letter", method=RequestMethod.GET)
    public @ResponseBody String trainNumber() {
		letterTrainer = new LetterTrainer();
		letterTrainer.generateTrainedModel(basePath + File.separator + "training", basePath + File.separator + "minist-model.zip");
		return "Training initiated. Trained model will be generated in few minutes!";
    }
}
