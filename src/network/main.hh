#include "tiny_dnn/tiny_dnn.h"

// Declaring the type of Predicate that accepts 2 pairs and return a bool
typedef std::function<bool(std::pair<int, double>, std::pair<int, double>)> Comparator;

// Defining a lambda function to compare two pairs. It will compare two pairs using second field
Comparator compareFunction =
[](std::pair<int, double> elem1, std::pair<int, double> elem2)
{
	return elem1.second < elem2.second;
};

// Functions declaration

template <typename Activation>
double rescale(double x);

float roundb(float f, int bits);

void convert_image(const std::string &imagefilename, double minv, double maxv, int w, int h, tiny_dnn::vec_t &data);

void single_file_test(const std::string &dictionary);

void NCA(tiny_dnn::network<tiny_dnn::sequential> &nn, int target);

static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn, tiny_dnn::core::backend_t backend_type, int a, int b);

static float training(const std::string &data_dir_path, double learning_rate, const int n_train_epochs, const int n_minibatch, tiny_dnn::core::backend_t backend_type);

static float training_approx(const std::string &dictionary, const std::string &data_dir_path, double learning_rate, const int n_train_epochs, const int n_minibatch, tiny_dnn::core::backend_t backend_type);

static float training_fs_approx(const std::string &data_dir_path, double learning_rate, const int n_train_epochs, const int n_minibatch, tiny_dnn::core::backend_t backend_type);

static void comparison(const std::string &dictionary, const std::string &data_dir_path, double learning_rate, const int n_train_epochs, const int n_minibatch, tiny_dnn::core::backend_t backend_type);

static void weights_approximation(tiny_dnn::network<tiny_dnn::sequential> &nn, int neuron_number, int bits);

void approximation(const std::string &dictionary);

float test_set_eval(const std::string &dictionary, const std::string &data_dir_path);

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name);

void print();

static void usage(const char *argv0);
