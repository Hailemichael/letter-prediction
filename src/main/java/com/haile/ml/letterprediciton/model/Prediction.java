package com.haile.ml.letterprediciton.model;

import java.math.BigDecimal;
import java.util.HashMap;

public class Prediction {
    private final Integer prediction;
    private final HashMap<String, BigDecimal> percentage;

    public Prediction(Integer prediction, HashMap<String, BigDecimal> percentage) {
        this.prediction = prediction;
        this.percentage = percentage;
    }

    public Integer getPrediction() {
        return prediction;
    }
    
    public HashMap<String, BigDecimal> getPercentage() {
        return percentage;
    }

}
