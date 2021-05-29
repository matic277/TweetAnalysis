import opennlp.tools.doccat.*;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class Main {

    static DoccatModel model;

    public static void main(String[] args) throws IOException {
        train();
        classify();
    }

    public static void classify() throws IOException {
        Files.lines(Paths.get("./data/text.txt")).forEach(sentence -> {
            DocumentCategorizerME myCategorizer = new DocumentCategorizerME(model);
            double[] outcomes = myCategorizer.categorize(sentence.split(" "));
            String category = myCategorizer.getBestCategory(outcomes);

            if (category.equalsIgnoreCase("1")) {
                System.out.println(" 1 => " + sentence);
            } else {
                System.out.println("-1 => " + sentence);
            }
        });
    }

    public static void train() {
        final InputStream dataIn;
        try {
            dataIn = new FileInputStream("./data/trainingSet.txt");

            ObjectStream<String> lineStream = new PlainTextByLineStream(() -> dataIn, "UTF-8");
            ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);

            TrainingParameters trainParams = new TrainingParameters();
            trainParams.put(TrainingParameters.CUTOFF_PARAM, 2);
            trainParams.put(TrainingParameters.ITERATIONS_PARAM, 30);
            trainParams.put(TrainingParameters.THREADS_PARAM, 2);

            model = DocumentCategorizerME.train(
                    "en",
                    sampleStream,
                    trainParams,
                    new DoccatFactory());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
