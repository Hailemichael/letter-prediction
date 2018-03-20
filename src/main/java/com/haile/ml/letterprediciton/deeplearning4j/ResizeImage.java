package com.haile.ml.letterprediciton.deeplearning4j;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ResizeImage {
	private static final Logger logger = LoggerFactory.getLogger(ResizeImage.class);
	private static final String basePath = "C:\\Users\\Haile\\Documents\\Haile\\machine-learning\\English\\Hnd";

	public static void main(String[] args) {

		File collectionOfFolders = new File(basePath + File.separator + "Img");
		File[] letterFolders = collectionOfFolders.listFiles();

		for (File folder : letterFolders) {
			String parentPath = folder.getName();
			File[] images = folder.listFiles();
			for (File image : images) {
				String fileName = image.getName();
				File file = new File(image.getAbsolutePath());
				BufferedImage originalImage = null;
				try {
					originalImage = ImageIO.read(file);
				} catch (IOException e) {
					logger.info("Exception while reading file: " + fileName);
				}

				// Resize
				BufferedImage resizedImage = null;
				// resize image if larger than 600 x 600
				if ((originalImage.getWidth() > 600) || (originalImage.getHeight() > 600)) {
					resizedImage = resizeImage(originalImage, 28, 28, RenderingHints.VALUE_INTERPOLATION_BICUBIC, true);
				} else {
					resizedImage = originalImage;
				}
				
				new File(basePath + File.separator + "resizedImg" + File.separator + "testing" + File.separator + parentPath).mkdirs();
				
				try {
					ImageIO.write(resizedImage, "png", new File(basePath + File.separator + "resizedImg" + File.separator + "testing" + File.separator + parentPath + File.separator + fileName));
				} catch (IOException e) {
					logger.info("Exception while writing file: " + fileName);
				}

			}

		}

	}

	public static BufferedImage resizeImage(BufferedImage img, int targetWidth, int targetHeight, Object hint,
			boolean higherQuality) {
		/*int type = (img.getTransparency() == Transparency.OPAQUE) ? BufferedImage.TYPE_INT_RGB
				: BufferedImage.TYPE_INT_ARGB;*/
		BufferedImage ret = (BufferedImage) img;
		int w, h;
		if (higherQuality) {
			// Use multi-step technique: start with original size, then
			// scale down in multiple passes with drawImage()
			// until the target size is reached
			w = img.getWidth();
			h = img.getHeight();
		} else {
			// Use one-step technique: scale directly from original
			// size to target size with a single drawImage() call
			w = targetWidth;
			h = targetHeight;
		}

		do {
			if (higherQuality && w > targetWidth) {
				w /= 2;
				if (w < targetWidth) {
					w = targetWidth;
				}
			}

			if (higherQuality && h > targetHeight) {
				h /= 2;
				if (h < targetHeight) {
					h = targetHeight;
				}
			}

			BufferedImage tmp = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);// It should be 'type' variable for colored images
			Graphics2D g2 = tmp.createGraphics();
			g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, hint);
			g2.drawImage(ret, 0, 0, w, h, null);
			g2.dispose();

			ret = tmp;
		} while (w != targetWidth || h != targetHeight);

		return ret;
	}
}
