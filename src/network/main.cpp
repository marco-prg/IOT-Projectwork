#include <iostream>
#include <string>
#include <map>
#include <cmath>

#include "main.hh"

std::map<int, int> ranking;

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
	Activation a(1);
	return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

// bit approximation function
float roundb(float f, int bits) {
	union {			// num.i and num.f are mapped on same bits
		int i;
		float f;
	} num;

	bits = 32 - bits;
	num.f = f;
	num.i = num.i + (1 << (bits - 1));		// round instead of truncate
	num.i = num.i & (-1 << bits);			// AND bitwise between mask and rounded value
	return num.f;
}

void convert_image(const std::string &imagefilename,
	double minv,
	double maxv,
	int w,
	int h,
	tiny_dnn::vec_t &data) {
	tiny_dnn::image<> img(imagefilename, tiny_dnn::image_type::grayscale);
	tiny_dnn::image<> resized = resize_image(img, w, h);

	// mnist dataset is "white on black", so negate required
	std::transform(resized.begin(), resized.end(), std::back_inserter(data),
		[=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

void single_file_test(const std::string &dictionary) {
	std::string src_filename;
	int target;

	std::cout << std::endl << "*** Single file test ***" << std::endl
		<< "Please specify image datapath from current directory, including filename (e.g. ../../data/[0-9].bmp): " << std::endl;
	std::cin >> src_filename;
	std::cout << "Please specify target class (0-9): " << std::endl;
	std::cin >> target;

	tiny_dnn::network<tiny_dnn::sequential> nn;
	nn.load(dictionary);

	// convert imagefile to vec_t
	tiny_dnn::vec_t data;
	convert_image(src_filename, -1.0, 1.0, 32, 32, data);

	// recognize
	auto res = nn.predict(data);

	// sort & print results
	std::vector<std::pair<double, int>> scores;

	for (int i = 0; i < 10; i++)
		scores.emplace_back(rescale<tiny_dnn::tanh_layer>(res[i]), i);
	sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());

	std::cout << std::endl
		<< "*** Prediction results ***" << std::endl;
	for (int i = 0; i < 10; i++)
		std::cout << scores[i].second << ": " << scores[i].first << std::endl;

	// Neuron criticality analysis
	NCA(nn, target);
}

void NCA(tiny_dnn::network<tiny_dnn::sequential> &nn, int target) {

	// target array initialization
	double target_array[10] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	target_array[target] = 1.0;

	// criticality data structure initialization
	std::map<int, double> criticality;

	// Neuron criticality analysis start
	tiny_dnn::vec_t inputs, outputs;
	int size = 0;
	double result;
	double sum = 0.0;

	// Evaluation for output layer
	outputs = (*(nn[6]->outputs()[0])).get_data()[0][0];
	inputs = (*(nn[6]->inputs()[0])).get_data()[0][0];

	for (unsigned int i = 0; i < outputs.size(); i++) {
		result = -(target_array[i] - outputs[i]) * (1.0 - tanh(inputs[i]) * tanh(inputs[i]));
		criticality.insert(std::pair<int, double>(i, result));
	}
	size = outputs.size();

	// Evaluation for hidden layer 2
	outputs = (*(nn[4]->outputs()[0])).get_data()[0][0];
	inputs = (*(nn[4]->inputs()[0])).get_data()[0][0];

	for (unsigned int i = 0; i < outputs.size(); i++) {

		for (unsigned int j = 0; j < (*(nn[5]->weights()[1])).size(); j++)
			sum += criticality.at(j) * (*(nn[5]->weights()[0]))[i + j * outputs.size()];

		result = -(1.0 - tanh(inputs[i]) * tanh(inputs[i])) * sum;
		criticality.insert(std::pair<int, double>(i + size, result));
		sum = 0.0;
	}
	size += outputs.size();

	// Evaluation for hidden layer 1
	outputs = (*(nn[2]->outputs()[0])).get_data()[0][0];
	inputs = (*(nn[2]->inputs()[0])).get_data()[0][0];

	for (unsigned int i = 0; i < outputs.size(); i++) {

		for (unsigned int j = 0; j < (*(nn[3]->weights()[1])).size(); j++)
			sum += criticality.at(j + (*(nn[5]->weights()[1])).size()) * (*(nn[3]->weights()[0]))[i + j * outputs.size()];

		result = -(1.0 - tanh(inputs[i])*tanh(inputs[i]))* sum;
		criticality.insert(std::pair<int, double>(i + size, result));
		sum = 0.0;
	}
	size = 0;

	std::set <std::pair<int, double>, Comparator> crit_set(criticality.begin(), criticality.end(), compareFunction);
	std::cout << std::endl
		<< "*** Neuron criticality analysis ranking ***" << std::endl
		<< "Neurons are increasingly numbered from top to bottom." << std::endl
		<< "Results are shown in ascending order of criticality." << std::endl;

	int i = 0;
	// Ranking map clear
	ranking.clear();

	// NCA ranking save
	for (const auto &x : crit_set) {
		std::cout << "Neuron " << x.first << ": " << x.second << std::endl;
		ranking.insert(std::pair<int, int>(i++, x.first));
	}

	std::cout << std::endl
		<< "*** Results rescaling ***" << std::endl;
	for (const auto &x : crit_set)
		std::cout << "Neuron " << x.first << ": " << rescale<tiny_dnn::tanh_layer>(x.second) << std::endl;
}

static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
	tiny_dnn::core::backend_t backend_type, int a, int b) {
	// S : sub-sampling
	// F : fully connected
	// clang-format off
	using fc = tiny_dnn::layers::fc;
	using conv = tiny_dnn::layers::conv;
	using ave_pool = tiny_dnn::layers::ave_pool;
	using tanh = tiny_dnn::activation::tanh;

	using tiny_dnn::core::connection_table;
	using padding = tiny_dnn::padding;

	nn << ave_pool(32, 32, 1, 2)   // S1, 1@32x32-in, 1@16x16-out
		<< fc(256, a, true, backend_type)	// F2, 256-in, a-out
		<< tanh()
		<< fc(a, b, true, backend_type)  // F3, a-in, b-out
		<< tanh()
		<< fc(b, 10, true, backend_type)  // F4, b-in, 10-out
		<< tanh();
}

static float training(const std::string &data_dir_path,
	double learning_rate,
	const int n_train_epochs,
	const int n_minibatch,
	tiny_dnn::core::backend_t backend_type) {

	std::cout << "Running with the following parameters:" << std::endl
		<< "Data path: " << data_dir_path << std::endl
		<< "Learning rate: " << learning_rate << std::endl
		<< "Minibatch size: " << n_minibatch << std::endl
		<< "Number of epochs: " << n_train_epochs << std::endl
		<< "Backend type: " << backend_type << std::endl
		<< std::endl;

	// specify loss-function and learning strategy
	tiny_dnn::network<tiny_dnn::sequential> nn;
	tiny_dnn::adagrad optimizer;
	int n1, n2;

	std::cout << "*** Network construction ***" << std::endl
		<< "Please specify the number of neurons for each layer." << std::endl
		<< "Layer 1: " << std::endl;
	std::cin >> n1;
	std::cout << "Layer 2: " << std::endl;
	std::cin >> n2;
	std::cout << "Layer 3: 10" << std::endl;

	construct_net(nn, backend_type, n1, n2);

	std::cout << "*** Loading dataset ***" << std::endl;

	// load MNIST dataset
	std::vector<tiny_dnn::label_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
		&train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
		&train_images, -1.0, 1.0, 2, 2);
	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
		&test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
		&test_images, -1.0, 1.0, 2, 2);

	std::cout << "*** Start training ***" << std::endl;

	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	optimizer.alpha *=
		std::min(tiny_dnn::float_t(4),
			static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {
		std::cout << std::endl << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;
		++epoch;
		tiny_dnn::result res = nn.test(test_images, test_labels);
		std::cout << "Test accuracy: " << (float)res.num_success / (float)res.num_total * 100 << "%" << std::endl;

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() {
		disp += n_minibatch;
	};

	// training
	nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, n_minibatch,
		n_train_epochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	std::cout << "End training." << std::endl;

	// test and show results
	tiny_dnn::result res = nn.test(test_images, test_labels);
	res.print_detail(std::cout);

	// save network model & trained weights
	nn.save("C-MLP-model");
	return res.accuracy();
}

static float training_approx(const std::string &dictionary,
	const std::string &data_dir_path,
	double learning_rate,
	const int n_train_epochs,
	const int n_minibatch,
	tiny_dnn::core::backend_t backend_type) {

	std::cout << std::endl << "*** Training approximated network ***" << std::endl;

	if (ranking.empty()) {
		std::cout << "No NCA ranking available. " << std::endl;
		return -1.0;
	}

	int neuron_number = 0;
	int bits = 0;
	int net_neuron_number = 0;

	tiny_dnn::network<tiny_dnn::sequential> nn;
	nn.load(dictionary);

	for (size_t i = 0; i < nn.depth(); i++) {
		if (nn[i]->layer_type() == "fully-connected") {
			net_neuron_number += (*(nn[i]->weights()[1])).size();
		}
	}

	while (neuron_number < 1) {
		std::cout << "Please specify the number of neurons to approximate (the network has a total of "
			<< net_neuron_number << " neurons): " << std::endl;
		std::cin >> neuron_number;
	}
	if (neuron_number > net_neuron_number)
		neuron_number = net_neuron_number;

	while (bits < 10 || bits > 31) {
		std::cout << "Please specify the number of bits to use (between 10 and 31): " << std::endl;
		std::cin >> bits;
	}

	weights_approximation(nn, neuron_number, bits);

	std::cout << "Training start with the following parameters:" << std::endl
		<< "Data path: " << data_dir_path << std::endl
		<< "Learning rate: " << learning_rate << std::endl
		<< "Minibatch size: " << n_minibatch << std::endl
		<< "Number of epochs: " << n_train_epochs << std::endl
		<< "Backend type: " << backend_type << std::endl
		<< std::endl;

	// specify loss-function and learning strategy
	tiny_dnn::adagrad optimizer;

	std::cout << "*** Loading dataset ***" << std::endl;

	// load MNIST dataset
	std::vector<tiny_dnn::label_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
		&train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
		&train_images, -1.0, 1.0, 2, 2);
	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
		&test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
		&test_images, -1.0, 1.0, 2, 2);

	std::cout << "*** Start training ***" << std::endl;

	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	optimizer.alpha *=
		std::min(tiny_dnn::float_t(4),
			static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {

		weights_approximation(nn, neuron_number, bits);

		std::cout << std::endl << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;
		++epoch;
		tiny_dnn::result res = nn.test(test_images, test_labels);
		std::cout << "Test accuracy: " << (float)res.num_success / (float)res.num_total * 100 << "%" << std::endl;

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() {
		disp += n_minibatch;
		weights_approximation(nn, neuron_number, bits);
	};

	// training
	nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, n_minibatch,
		n_train_epochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	weights_approximation(nn, neuron_number, bits);

	std::cout << "End training." << std::endl;

	// test and show results
	tiny_dnn::result res = nn.test(test_images, test_labels);
	res.print_detail(std::cout);

	// save network model & trained weights
	nn.save("C-MLP-model");
	return res.accuracy();
}

static float training_fs_approx(const std::string &data_dir_path,
	double learning_rate,
	const int n_train_epochs,
	const int n_minibatch,
	tiny_dnn::core::backend_t backend_type) {

	std::cout << std::endl << "*** Training approximated network from scratch ***" << std::endl;

	if (ranking.empty()) {
		std::cout << "No NCA ranking available. " << std::endl;
		return -1.0;
	}

	std::cout << "Training start with the following parameters:" << std::endl
		<< "Data path: " << data_dir_path << std::endl
		<< "Learning rate: " << learning_rate << std::endl
		<< "Minibatch size: " << n_minibatch << std::endl
		<< "Number of epochs: " << n_train_epochs << std::endl
		<< "Backend type: " << backend_type << std::endl
		<< std::endl;

	// specify loss-function and learning strategy
	tiny_dnn::network<tiny_dnn::sequential> nn;
	tiny_dnn::adagrad optimizer;
	int n1, n2;

	std::cout << "*** Network construction ***" << std::endl
		<< "Please specify the number of neurons for each layer." << std::endl
		<< "Layer 1: " << std::endl;
	std::cin >> n1;
	std::cout << "Layer 2: " << std::endl;
	std::cin >> n2;
	std::cout << "Layer 3: 10" << std::endl;

	construct_net(nn, backend_type, n1, n2);

	int neuron_number = 0;
	int bits = 0;
	int net_neuron_number = n1 + n2 + 10;

	std::cout << std::endl << "*** Network approximation ***" << std::endl;

	while (neuron_number < 1) {
		std::cout << "Please specify the number of neurons to approximate (the network has a total of "
			<< net_neuron_number << " neurons): " << std::endl;
		std::cin >> neuron_number;
	}
	if (neuron_number > net_neuron_number)
		neuron_number = net_neuron_number;

	while (bits < 10 || bits > 31) {
		std::cout << "Please specify the number of bits to use (between 10 and 31): " << std::endl;
		std::cin >> bits;
	}

	weights_approximation(nn, neuron_number, bits);

	std::cout << "*** Loading dataset ***" << std::endl;

	// load MNIST dataset
	std::vector<tiny_dnn::label_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
		&train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
		&train_images, -1.0, 1.0, 2, 2);
	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
		&test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
		&test_images, -1.0, 1.0, 2, 2);

	std::cout << "*** Start training ***" << std::endl;

	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	optimizer.alpha *=
		std::min(tiny_dnn::float_t(4),
			static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]() {

		weights_approximation(nn, neuron_number, bits);

		std::cout << std::endl << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
			<< t.elapsed() << "s elapsed." << std::endl;
		++epoch;
		tiny_dnn::result res = nn.test(test_images, test_labels);
		std::cout << "Test accuracy: " << (float)res.num_success / (float)res.num_total * 100 << "%" << std::endl;

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() {
		disp += n_minibatch;
		weights_approximation(nn, neuron_number, bits);
	};

	// training
	nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, n_minibatch,
		n_train_epochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	weights_approximation(nn, neuron_number, bits);

	std::cout << "End training." << std::endl;

	// test and show results
	tiny_dnn::result res = nn.test(test_images, test_labels);
	res.print_detail(std::cout);

	// save network model & trained weights
	nn.save("C-MLP-model");
	return res.accuracy();
}

static void comparison(const std::string &dictionary,
	const std::string &data_dir_path,
	double learning_rate,
	const int n_train_epochs,
	const int n_minibatch,
	tiny_dnn::core::backend_t backend_type) {

	float accuracy[4] = { -1.0, -1.0, -1.0, -1.0 };

	try {
		accuracy[0] = training(data_dir_path, learning_rate, n_train_epochs, n_minibatch, backend_type);
	}
	catch (tiny_dnn::nn_error &err) {
		std::cerr << "Exception: " << err.what() << std::endl;
		return;
	}
	single_file_test(dictionary);
	approximation(dictionary);
	accuracy[1] = test_set_eval(dictionary, data_dir_path);
	try {
		accuracy[2] = training_approx(dictionary, data_dir_path, learning_rate, n_train_epochs, n_minibatch, backend_type);
	}
	catch (tiny_dnn::nn_error &err) {
		std::cerr << "Exception: " << err.what() << std::endl;
		return;
	}
	try {
		accuracy[3] = training_fs_approx(data_dir_path, learning_rate, n_train_epochs, n_minibatch, backend_type);
	}
	catch (tiny_dnn::nn_error &err) {
		std::cerr << "Exception: " << err.what() << std::endl;
		return;
	}

	std::cout << std::endl << "*** Full comparison final results ***" << std::endl
		<< "Original configuration network (Aorig): Accuracy (after training) = " << accuracy[0] << std::endl
		<< "Original network with approximated weights (Aapprox): Accuracy (test only) = " << accuracy[1] << std::endl
		<< "Pre-trained approximated network (Aapprox2): Accuracy (after fine-tuning) = " << accuracy[2] << std::endl
		<< "Approximated network from scratch (Aapprox3): Accuracy (after training) = " << accuracy[3] << std::endl;
}

static void weights_approximation(tiny_dnn::network<tiny_dnn::sequential> &nn,
	int neuron_number, int bits) {

	int l1, l2, l3, layer = 0, offset = 0;
	l1 = (*(nn[5]->weights()[1])).size();
	l2 = (*(nn[3]->weights()[1])).size();
	l3 = (*(nn[1]->weights()[1])).size();

	for (const auto &x : ranking) {
		if (x.first >= neuron_number)
			return;

		//std::cout << x.first << ") Approximation of neuron number " << x.second << ". " << std::endl;

		if (x.second < (l1 + l2 + l3)) {
			layer = 1;
			offset = x.second - l1 - l2;
		}
		if (x.second < (l1 + l2)) {
			layer = 3;
			offset = x.second - l1;
		}
		if (x.second < l1) {
			layer = 5;
			offset = x.second;
		}

		tiny_dnn::vec_t & weights = *(nn[layer]->weights()[0]);
		for (unsigned int i = 0; i < nn[layer]->in_size(); i++) {
			//std::cout << (i + offset * net[layer]->in_size()) << " " << i << " " << offset << " "
			//	<< net[layer]->in_size() << " " << weights[i + offset * net[layer]->in_size()] << " ";
			weights[i + offset * nn[layer]->in_size()] = roundb(weights[i + offset * nn[layer]->in_size()], bits);
			//std::cout << weights[i + offset * net[layer]->in_size()] << std::endl;
		}
	}
}

void approximation(const std::string &dictionary) {

	std::cout << std::endl << "*** Weights approximation ***" << std::endl;

	if (ranking.empty()) {
		std::cout << "No NCA ranking available. " << std::endl;
		return;
	}

	int neuron_number = 0;
	int bits = 0;
	int net_neuron_number = 0;

	tiny_dnn::network<tiny_dnn::sequential> nn;
	nn.load(dictionary);

	for (size_t i = 0; i < nn.depth(); i++) {
		if (nn[i]->layer_type() == "fully-connected") {
			net_neuron_number += (*(nn[i]->weights()[1])).size();
		}
	}

	while (neuron_number < 1) {
		std::cout << "Please specify the number of neurons to approximate (the network has a total of "
			<< net_neuron_number << " neurons): " << std::endl;
		std::cin >> neuron_number;
	}
	if (neuron_number > net_neuron_number)
		neuron_number = net_neuron_number;

	while (bits < 10 || bits > 31) {
		std::cout << "Please specify the number of bits to use (between 10 and 31): " << std::endl;
		std::cin >> bits;
	}

	weights_approximation(nn, neuron_number, bits);
	nn.save("C-MLP-model");
}

float test_set_eval(const std::string &dictionary, const std::string &data_dir_path) {
	tiny_dnn::network<tiny_dnn::sequential> nn;
	nn.load(dictionary);

	// load MNIST dataset
	std::vector<tiny_dnn::label_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
		&train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
		&train_images, -1.0, 1.0, 2, 2);
	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
		&test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
		&test_images, -1.0, 1.0, 2, 2);

	// test and show results
	std::cout << "*** Start test set evaluation ***" << std::endl;
	tiny_dnn::result res = nn.test(test_images, test_labels);
	std::cout << "Results = ";
	res.print_detail(std::cout);	
	return res.accuracy();
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
	const std::array<const std::string, 5> names = { {
	  "internal", "nnpack", "libdnn", "avx", "opencl",
	} };
	for (size_t i = 0; i < names.size(); ++i) {
		if (name.compare(names[i]) == 0) {
			return static_cast<tiny_dnn::core::backend_t>(i);
		}
	}
	return tiny_dnn::core::default_engine();
}

void print() {
	tiny_dnn::network<tiny_dnn::sequential> nn;
	nn.load("C-MLP-model");
	int index = 3, n = 0;

	for (size_t k = nn.depth()-1 ; k > 0; k--) {
		if (nn[k]->layer_type() == "fully-connected") {
			std::cout << "*** Layer " << index-- << " weights *** " << std::endl;
			tiny_dnn::vec_t & weights = *(nn[k]->weights()[0]);
			for (size_t j = 0; j < (*(nn[k]->weights()[1])).size(); j++) {

				std::cout << "Neuron " << n++ << " weights: " << std::endl;
				for (size_t i = 0; i < nn[k]->in_size(); i++)
					std::cout << weights[i + j * nn[k]->in_size()] << ' ';
				std::cout << std::endl;
			}
			std::cout << "Size = " << weights.size() << std::endl << std::endl;
		}
	}
}

static void usage(const char *argv0) {
	std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
		<< " --learning_rate 0.1"
		<< " --epochs 15"
		<< " --minibatch_size 64"
		<< " --backend_type internal" << std::endl;
}

int main(int argc, char **argv) {
	double learning_rate = 0.1;
	int epochs = 5;
	std::string data_path = "../../data";
	int minibatch_size = 64;
	tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

	if (argc == 2) {
		std::string argname(argv[1]);
		if (argname == "--help" || argname == "-h") {
			usage(argv[0]);
			return 0;
		}
	}
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--learning_rate") {
			learning_rate = atof(argv[count + 1]);
		}
		else if (argname == "--epochs") {
			epochs = atoi(argv[count + 1]);
		}
		else if (argname == "--minibatch_size") {
			minibatch_size = atoi(argv[count + 1]);
		}
		else if (argname == "--backend_type") {
			backend_type = parse_backend_name(argv[count + 1]);
		}
		else if (argname == "--data_path") {
			data_path = std::string(argv[count + 1]);
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			usage(argv[0]);
			return -1;
		}
	}
	if (learning_rate <= 0) {
		std::cerr
			<< "Invalid learning rate. The learning rate must be greater than 0."
			<< std::endl;
		return -1;
	}
	if (epochs <= 0) {
		std::cerr << "Invalid number of epochs. The number of epochs must be greater than 0." << std::endl;
		return -1;
	}
	if (minibatch_size <= 0 || minibatch_size > 60000) {
		std::cerr
			<< "Invalid minibatch size. The minibatch size must be greater than 0 and less than dataset size (60000)."
			<< std::endl;
		return -1;
	}

	int command = 0;
	while (true) {
		std::cout << std::endl
			<< " *** Configurable Multilayer Perceptron for Neuron Criticality Analysis and Approximate Computing *** " << std::endl
			<< std::endl
			<< "Please digit one of the following command number:" << std::endl
			<< "1) Network construction and training using original configuration" << std::endl
			<< "2) Single file input test and Neuron Criticality Analysis (NCA)" << std::endl
			<< "3) Network weights approximation (based on last NCA ranking)" << std::endl
			<< "4) Complete test set evaluation" << std::endl
			<< "5) Training existent model with weights approximation (based on last NCA ranking)" << std::endl
			<< "6) Training new model from scratch with weights approximation (based on last NCA ranking)" << std::endl
			<< "7) Full comparison (Aorig, Aapprox, Aapprox2, Aapprox3)" << std::endl
			<< "8) Print network weights (most recent configuration)" << std::endl
			<< "9) Exit" << std::endl
			<< std::endl;

		std::cin >> command;

		if (!std::cin) {
			std::cout << "Invalid command" << std::endl;
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			continue;
		}

		switch (command) {
		case 1:
			try {
				training(data_path, learning_rate, epochs, minibatch_size, backend_type);
			}
			catch (tiny_dnn::nn_error &err) {
				std::cerr << "Exception: " << err.what() << std::endl;
				return 0;
			}
			break;
		case 2:
			single_file_test("C-MLP-model");
			break;
		case 3:
			approximation("C-MLP-model");
			break;
		case 4:
			test_set_eval("C-MLP-model", data_path);
			break;
		case 5:
			try {
				training_approx("C-MLP-model", data_path, learning_rate, epochs, minibatch_size, backend_type);
			}
			catch (tiny_dnn::nn_error &err) {
				std::cerr << "Exception: " << err.what() << std::endl;
				return 0;
			}
			break;
		case 6:
			try {
				training_fs_approx(data_path, learning_rate, epochs, minibatch_size, backend_type);
			}
			catch (tiny_dnn::nn_error &err) {
				std::cerr << "Exception: " << err.what() << std::endl;
				return 0;
			}
			break;
		case 7:
			comparison("C-MLP-model", data_path, learning_rate, epochs, minibatch_size, backend_type);
			break;
		case 8:
			print();
			break;
		case 9:
			return 0;
			break;
		default:
			std::cout << "Invalid command number" << std::endl;
			break;
		}
	}
}