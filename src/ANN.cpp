#include "../include/ANN.h"

ANN::ANN(std::vector<int> layer_sizes_n, std::vector<Activation> activations_n,
         unsigned int seed)
    : layer_sizes(std::move(layer_sizes_n)),
      activations(std::move(activations_n)) {
  if (layer_sizes.size() != activations.size() + 1) {
    throw std::invalid_argument("Layer counts must match activation counts!");
  }

  if (layer_sizes.size() == 0) {
    throw std::invalid_argument("Cannot have empty ANN!");
  }

  // Can change in future if we want to generalize this class.
  for (size_t i = 0; i < layer_sizes.size(); ++i) {
    if (i < layer_sizes.size() - 2 && activations[i] == SOFTMAX) {
      throw std::invalid_argument("Hidden layers cannot have SOFTMAX.");
    }
    if (i == layer_sizes.size() - 2 && activations[i] != SOFTMAX) {
      throw std::invalid_argument(
          "Final layer must have SOFTMAX for cross_entropy backprop.");
    }
  }

  int prev_size = layer_sizes[0];
  for (size_t i = 1; i < layer_sizes.size(); ++i) {
    int curr_size = layer_sizes[i];
    int dims_weights[] = {prev_size, curr_size};
    int dims_biases[] = {curr_size};
    weights.push_back(Tensor(dims_weights, 2, seed));
    biases.push_back(Tensor(dims_biases, 1, (float)0));
    prev_size = curr_size;
  }
}

Tensor ANN::forward(const Tensor& input) {

  //Compute a(i) = activation(z(i))
  //Compute z(i) = a(i-1) * w(i) + b(i)

  if (input.get_dim_len() != 2 || input.get_dims_index(1) != layer_sizes[0]) {
    throw std::invalid_argument("Invalid input size!");
  }

  z_cache.clear();
  a_cache.clear();

  Tensor ret = input.clone();
  for (size_t i = 0; i < weights.size(); ++i) {
    a_cache.push_back(ret.clone());
    ret = ret.matmul(weights[i]);
    if (i != weights.size() - 1) {
      z_cache.push_back(ret.clone());
    }
    ret.add_inplace(biases[i]); //Shared across all samples
    if (activations[i] == RELU) {
      ret.apply_inplace(relu);
    } else if (activations[i] == SIGMOID) {
      ret.apply_inplace(sigmoid);
    } else if (activations[i] == SOFTMAX) {
      apply_stable_softmax(ret);
    } else {
      throw std::invalid_argument("Invalid activation.");
    }
  }

  return ret;
}

void ANN::backprop(const Tensor& init_d_loss) {

  Tensor d_loss = init_d_loss.clone(); //Error term dC/dz(i-1) =  da/dz * dC/dz, batch_size x curr_size (initialized as output size)
  for (int i = weights.size() - 1; i >= 0; --i) {

    //dC/dw = dz/dw * da/dz * dC/dz = a(i-1)^T * error term
    grad_weights.push_back(a_cache[i].transpose2d().matmul(d_loss)); //Transpose since a will be batch_size x weight_size initially
    //dC/db = dz/db * da/dz * dC/dz = 1 * error term, sum because biases shared across all samples
    grad_biases.push_back(d_loss.sum_rows());     //Will be be weight_size (1d)

    if (i != 0) { //Compute next error term (dC/dz(i-1) = da(i-1)/dz(i-1)) * dz(i)/da(i-1) * dC/dz(i) (previous error term)

      //Weights are prev_size x curr_size, transpose so they are curr_size x prev_size - Error term dimensions now correct
      d_loss = d_loss.matmul(weights[i].transpose2d()); //dz(i)/da(i-1)
      if (activations[i - 1] == RELU) { //da(i-1)/dz(i-1)) (Derivitive of activation);
        //Multiply with previous error term dC/dz(i)
        d_loss.emul_inplace(z_cache[i - 1].apply(deriv_relu));
      } else if (activations[i - 1] == SIGMOID) {
        d_loss.emul_inplace(z_cache[i - 1].apply(deriv_sigmoid));
      } else {
        throw std::invalid_argument("Invalid activation for backprop.");
      }
    }
  }
}

void ANN::update_weights_biases() {
  if (grad_weights.size() != weights.size() ||
      grad_biases.size() != biases.size()) {
    throw std::invalid_argument("Invalid input size!");
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i].add_inplace(
        grad_weights[weights.size() - 1 - i].scmul(-learning_rate));
    biases[i].add_inplace(
        grad_biases[biases.size() - 1 - i].scmul(-learning_rate));
  }
}

void ANN::clear_grads_and_cache() {
  z_cache.clear();
  a_cache.clear();
  grad_weights.clear();
  grad_biases.clear();
}

float ANN::get_learning_rate() const { return learning_rate; }

void ANN::set_learning_rate(float l) { learning_rate = l; }

// Only for a 1D vectors - 1 x N.
float ANN::cross_entropy(const Tensor& truth, const Tensor& preds) {
  if (truth.get_dim_len() == 2 && preds.get_dim_len() == 2) {
    if (truth.get_dims_index(1) == preds.get_dims_index(1)) {
      float tot_loss = 0;
      for (int r = 0; r < preds.get_dims_index(0); ++r) {
        constexpr float epsilon = 1e-7;
        for (int i = 0; i < truth.get_dims_index(1); ++i) {
          tot_loss -=
              truth.get({r, i}) * logf(std::max(preds.get({r, i}), epsilon));
        }
      }
      return tot_loss / preds.get_dims_index(0);
    } else {
      throw std::invalid_argument(
          "Incompatible shapes for cross_entropy loss!");
    }
  } else {
    throw std::invalid_argument("Invalid dimensions for cross_entropy loss!");
  }
}

int ANN::sum_correct(const Tensor& truth, const Tensor& preds) {
  if (truth.get_dim_len() == 2 && preds.get_dim_len() == 2) {
    if (truth.get_dims_index(1) == preds.get_dims_index(1)) {
      int tot = 0;
      for (int r = 0; r < preds.get_dims_index(0); ++r) {
        int pred_max = 0;
        for (int i = 1; i < preds.get_dims_index(1); ++i) {
          if (preds.get({r, pred_max}) < preds.get({r, i})) {
            pred_max = i;
          }
        }
        for (int i = 0; i < truth.get_dims_index(1); ++i) {
          if (abs(1 - truth.get({r, i})) < 1e-7) {
            if (i == pred_max) {
              tot += 1;
            }
          }
        }
      }
      return tot;
    } else {
      throw std::invalid_argument("Incompatible shapes for correctness!");
    }
  } else {
    throw std::invalid_argument("Invalid dimensions for correctness!");
  }
}

// Only for a 1D vector. - 1 x N.
void ANN::apply_stable_softmax(Tensor& m) {
  if (m.get_dim_len() == 2) {
    for (int r = 0; r < m.get_dims_index(0); ++r) {
      float max = -INFINITY;
      for (int i = 0; i < m.get_dims_index(1); ++i) {
        float val = m.get({r, i});
        if (val > max) {
          max = val;
        }
      }
      float sum = 0;
      for (int i = 0; i < m.get_dims_index(1); ++i) {
        float val = expf(m.get({r, i}) - max);
        sum += val;
        m.set({r, i}, val);
      }
      for (int i = 0; i < m.get_dims_index(1); ++i) {
        m.set({r, i}, m.get({r, i}) / sum);
      }
    }
  } else {
    throw std::invalid_argument("Invalid dimension for softmax!");
  }
}

void ANN::print_weights(int size) const {
  for (size_t i = 0; i < weights.size(); ++i) {
    std::cout << "=== WEIGHT " << i << " ===\n";
    weights[i].print_data(size);
  }
}

void ANN::print_biases(int size) const {
  for (size_t i = 0; i < biases.size(); ++i) {
    std::cout << "=== BIAS " << i << " ===\n";
    biases[i].print_data(size);
  }
}

float ANN::relu(float x) { return std::max(0.0f, x); }

float ANN::sigmoid(float x) { return (1 / (1 + std::exp(-x))); }

float ANN::deriv_relu(float x) {
  if (x <= 0) {
    return 0;
  }
  return 1;
}

float ANN::deriv_sigmoid(float x) {
  float sig = sigmoid(x);
  return sig * (1 - sig);
}