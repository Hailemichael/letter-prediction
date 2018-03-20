package com.haile.ml.letterprediciton;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableAsync
public class LetterPredicitonApplication {

	public static void main(String[] args) {
		SpringApplication.run(LetterPredicitonApplication.class, args);
	}

}
