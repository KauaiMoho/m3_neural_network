#ifndef ANN_H
#define ANN_H
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "Tensor.h"

class ANN {
 public:
  enum Activation { RELU, SIGMOID, SOFTMAX };

  ANN(std::vector<int> layer_sizes_n, std::vector<Activation> activations_n,
      unsigned int seed = 0);
  Tensor forward(const Tensor& input);
  void backprop(const Tensor& init_d_loss);
  void update_weights_biases();
  void clear_grads_and_cache();

  float get_learning_rate() const;
  void set_learning_rate(float l);

  void print_weights(int size = 100) const;
  void print_biases(int size = 100) const;

  static void apply_stable_softmax(Tensor& m);
  static float cross_entropy(const Tensor& truth, const Tensor& preds);
  static int sum_correct(const Tensor& truth, const Tensor& preds);
  static float relu(float x);
  static float sigmoid(float x);
  static float deriv_relu(float x);
  static float deriv_sigmoid(float x);

 private:
  std::vector<int> layer_sizes;
  std::vector<Activation> activations;
  std::vector<Tensor> weights;
  std::vector<Tensor> biases;
  std::vector<Tensor> z_cache;
  std::vector<Tensor> a_cache;
  std::vector<Tensor> grad_weights;
  std::vector<Tensor> grad_biases;
  float learning_rate = 0.001;
};

#endif