#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <vector>
#include <pthread.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>

#include "../include/ANN.h"     // this contains the ANN class definition
#include "../include/Tensor.h"  // this contains the Tensor class definition

// Check compiled assembly for matrix class: g++ -O3 -S src/Tensor.cpp -o
// matmul.s -I include Check compiled assembly for unaligned matrix class: g++
// -O3 -mno-unaligned-access -S src/Tensor.cpp -o matmul_una.s -I include Check
// diff: diff -u matmul.s matmul_una.s | less

// X, Y
std::tuple<std::vector<Tensor>, std::vector<Tensor>> load_MNIST_data(
    bool train, int batch_size) {
  std::vector<Tensor> X;
  std::vector<Tensor> Y;

  std::ifstream imgFile;
  std::ifstream labFile;
  int images = 0;
  if (train) {
    imgFile = std::ifstream(
        "/Users/karmaluitel/Desktop/Github/m3_neural_network/data/"
        "train-images.idx3-ubyte",
        std::ios::binary);
    labFile = std::ifstream(
        "/Users/karmaluitel/Desktop/Github/m3_neural_network/data/"
        "train-labels.idx1-ubyte",
        std::ios::binary);
    images = 60000;
  } else {
    imgFile = std::ifstream(
        "/Users/karmaluitel/Desktop/Github/m3_neural_network/data/"
        "t10k-images.idx3-ubyte",
        std::ios::binary);
    labFile = std::ifstream(
        "/Users/karmaluitel/Desktop/Github/m3_neural_network/data/"
        "t10k-labels.idx1-ubyte",
        std::ios::binary);
    images = 10000;
  }

  if (!imgFile.is_open() || !labFile.is_open()) {
    throw std::runtime_error("Failed to open MNIST files");
  }

  // Skip Headers
  imgFile.seekg(16, std::ios::beg);
  labFile.seekg(8, std::ios::beg);

  constexpr int rows = 28;
  constexpr int cols = 28;

  int img_dims[2] = {batch_size, rows * cols};
  int lab_dims[2] = {batch_size, 10};

  // Ignore last few data entries that would be less than batch size.
  for (int ic = 0; ic + batch_size < images; ic += batch_size) {
    float* image_data =
        (float*)malloc(batch_size * rows * cols * sizeof(float));

    if (image_data == nullptr) {
      throw std::runtime_error("Memory allocation error");
    }

    float* label_data = (float*)malloc(batch_size * 10 * sizeof(float));

    if (image_data == nullptr) {
      throw std::runtime_error("Memory allocation error");
    }

    for (int i = 0; i < batch_size * 10; ++i) {
      label_data[i] = 0;
    }

    for (int i = 0; i < batch_size; ++i) {
      std::vector<unsigned char> temp_img(rows * cols);
      imgFile.read((char*)temp_img.data(), rows * cols);
      for (int j = 0; j < rows * cols; ++j) {
        image_data[(i * rows * cols) + j] =
            static_cast<float>(temp_img[j]) /
            255.0f;  // Normalize image intensity.
      }

      unsigned char temp_label;
      labFile.read((char*)&temp_label, sizeof(temp_label));
      label_data[(i * 10) + (int)temp_label] = 1.0f;
    }

    X.push_back(Tensor(img_dims, 2, image_data));
    Y.push_back(Tensor(lab_dims, 2, label_data));

    free(image_data);
    free(label_data);
  }

  return std::make_tuple(X, Y);
}

void train(ANN& model, const std::vector<Tensor>& X,
           const std::vector<Tensor>& Y, int epochs, int batch_size) {
  for (size_t i = 0; i < epochs; ++i) {
    int correct = 0;
    float loss = 0;
    for (size_t i = 0; i < X.size(); ++i) {
      Tensor output = model.forward(X[i]);
      loss += ANN::cross_entropy(Y[i], output);
      correct += ANN::sum_correct(Y[i], output);
      Tensor d_loss = output.add(Y[i].scmul(-1)).scmul(1.0 / batch_size);
      model.backprop(d_loss);
      model.update_weights_biases();
      model.clear_grads_and_cache();
    }
    std::cout << "Epoch: " << i << " === Loss: " << loss / (float)X.size()
              << " === Accuracy: " << correct / ((float)X.size() * batch_size)
              << "\n";
  }
}

void test(ANN& model, const std::vector<Tensor>& X,
          const std::vector<Tensor>& Y, int batch_size) {
  int correct = 0;
  for (size_t i = 0; i < X.size(); ++i) {
    Tensor test_output = model.forward(X[i]);
    model.clear_grads_and_cache();
    correct += ANN::sum_correct(Y[i], test_output);
  }
  std::cout << "Test accuracy: " << correct / ((float)X.size() * batch_size)
            << "\n";
}

int main() {
  pthread_t t = pthread_self();
  qos_class_t qos;
  int ret = pthread_get_qos_class_np(t, &qos, NULL);
  std::cout << ret << "\n";
  
  std::vector<int> sizes = {784, 128, 64, 10};
  std::vector<ANN::Activation> activations = {
      ANN::Activation::RELU, ANN::Activation::RELU, ANN::Activation::SOFTMAX};
  ANN model = ANN(sizes, activations, 17);

  model.print_weights();
  model.print_biases();
  std::cout << "\n";

  constexpr int batch_size = 64;

  std::tuple<std::vector<Tensor>, std::vector<Tensor>> train_dataset =
      load_MNIST_data(true, batch_size);
  std::tuple<std::vector<Tensor>, std::vector<Tensor>> test_dataset =
      load_MNIST_data(false, batch_size);

  auto start = std::chrono::high_resolution_clock::now();

  train(model, std::get<0>(train_dataset), std::get<1>(train_dataset), 5,
        batch_size);
  test(model, std::get<0>(test_dataset), std::get<1>(test_dataset), batch_size);

  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "\n";
  model.print_weights();
  model.print_biases();
  std::cout << "\n";

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Time elapsed: " << duration.count() / 1000.0 << " s\n";

  return 0;
}
