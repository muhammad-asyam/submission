const tf = require("@tensorflow/tfjs-node");

async function predictClassification(model, image) {
    try {
      const tensor = tf.node
        .decodeJpeg(image)
        .resizeNearestNeighbor([224, 224])
        .expandDims()
        .toFloat();

      const prediction = model.predict(tensor);
      const classes = [
        "Melanocytic nevus",
        "Squamous cell carcinoma",
        "Vascular lesion",
      ];

      const classResult = tf.argMax(prediction, 1).dataSync()[0];
      const label = classes[classResult];

      let suggestion;

      if (label === "Melanocytic nevus") {
        suggestion =
          "Segera konsultasi dengan dokter terdekat jika ukuran semakin membesar dengan cepat, mudah luka atau berdarah.";
      }

      if (label === "Squamous cell carcinoma") {
        suggestion =
          "Segera konsultasi dengan dokter terdekat untuk meminimalisasi penyebaran kanker.";
      }

      if (label === "Vascular lesion") {
        suggestion =
          "Segera konsultasi dengan dokter terdekat untuk mengetahui detail terkait tingkat bahaya penyakit.";
      }
      return {label, suggestion };
    } catch (error) {
      throw new InputError(`Terjadi kesalahan input: ${error.message}`);
    }
}

module.exports = predictClassification;
