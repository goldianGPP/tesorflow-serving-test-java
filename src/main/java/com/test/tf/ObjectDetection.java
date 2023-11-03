package com.test.tf;

import org.tensorflow.*;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.core.Slice;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.io.ReadFile;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

import java.util.HashMap;
import java.util.Map;

public class ObjectDetection {

    private static final String OD_MODEL_PATH = "C:/Users/USER/Documents/Teravin/DST/bin/model/DstHologramDetection_v2/";

    private final static String[] cocoLabels = new String[]{
            "background",
            "chip",
            "island",
            "new_hologram",
            "old_hologram",
            "qr"
    };

    public Tensor detect(String imagePath) {
        SavedModelBundle model = SavedModelBundle.load(OD_MODEL_PATH, "serve");

        try (Graph graph = new Graph(); Session session = new Session(graph)) {
            Ops tf = Ops.create(graph);
            Constant<TString> fileName = tf.constant(imagePath);
            ReadFile readFile = tf.io.readFile(fileName);

            DecodeJpeg.Options options = DecodeJpeg.channels(3L);
            DecodeJpeg decodeImage = tf.image.decodeJpeg(readFile.contents(), options);

            Shape imageShape = session.runner().fetch(decodeImage).run().get(0).shape();

            Reshape<TUint8> reshape = tf.reshape(decodeImage,
                    tf.array(1,
                            imageShape.asArray()[0],
                            imageShape.asArray()[1],
                            imageShape.asArray()[2]
                    )
            );
            try (TUint8 reshapeTensor = (TUint8) session.runner().fetch(reshape).run().get(0)) {

                model.signatures().listIterator().forEachRemaining(System.out::println);

                Map<String, Tensor> feedDict = new HashMap<>();
                feedDict.put("input_tensor", reshapeTensor);
                Result outputTensorMap = model.function("serving_default").call(feedDict);

                try (TFloat32 detectionBoxes = (TFloat32) outputTensorMap.get("detection_boxes").get();
                     TFloat32 detectionClasses = (TFloat32) outputTensorMap.get("detection_classes").get();
                     TFloat32 detectionScores = (TFloat32) outputTensorMap.get("detection_scores").get()
                ) {

                    int detectionIndex = argmax(detectionClasses);

                    float detectionScore = detectionScores.getFloat(0, detectionIndex);
                    System.out.println("detectionScores : " + detectionScore);
                    System.out.println("detectionClass : " + cocoLabels[((int) detectionClasses.getFloat(0, detectionIndex) - 1)]);

                    float ymin = detectionBoxes.get(0, detectionIndex).getFloat(0);
                    float xmin = detectionBoxes.get(0, detectionIndex).getFloat(1);
                    float ymax = detectionBoxes.get(0, detectionIndex).getFloat(2);
                    float xmax = detectionBoxes.get(0, detectionIndex).getFloat(3);

                    System.out.printf("ymin: %f%n", ymin);
                    System.out.printf("xmin: %f%n", xmin);
                    System.out.printf("ymax: %f%n", ymax);
                    System.out.printf("xmax: %f%n", xmax);

                    int imageWidth = (int) imageShape.asArray()[1];
                    int imageHeight = (int) imageShape.asArray()[0];
                    int left = (int) (xmin * imageWidth);
                    int top = (int) (ymin * imageHeight);
                    int right = (int) (xmax * imageWidth);
                    int bottom = (int) (ymax * imageHeight);

                    Slice<TUint8> croppedImage = tf.slice(
                            tf.constant(reshapeTensor),
                            tf.constant(new int[]{0, top, left, 0}),
                            tf.constant(new int[]{1, bottom - top, right - left, 3})
                    );

                    Tensor result = session.runner().fetch(croppedImage).run().get(0);

                    model.close();
                    return result;

                }
            }
        }
    }

    private int argmax(TFloat32 detectionClasses) {
        for (int i = 0; i < detectionClasses.get(0).size(); i++) {
            if (cocoLabels[((int) detectionClasses.getFloat(0, i) - 1)].equalsIgnoreCase("new_hologram")){
                return i;
            }
        }
        return 0;
    }
}
