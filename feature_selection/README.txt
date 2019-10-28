Basic workflow:
  1) Run the run_selector.py file to select the wavelengths. Note, for the
     genetic algorithm, this returns the entire population and the associated
     scores. Additional processing using gaussian_bandwidth.py must be
     performed to extract the HAGRID based wavelengths. run_selector.py
     writes files with the GA population and scores to a separate file for
     further processing.

     Make sure the write flag is set to True in order to produce files.

     For the run_ga_selector method:
       run_selector.py calls Selector class in wavelength_selection.py, which
       in turn calls the GA class from genetic_algorithm.py, which in turn
       uses one of the various Classifier classes in classifiers.py

  2) Run final_classify.py.

     final_classify.py imports the Transformer and Selector classes from
     transformer.py and selector.py, respectively. Transformer is to
     transform the original dataset (with all wavelengths present) to a new
     dateset that mimics data captured by a multispectral camera, then passes
     the transformed data to the MLP class from mlp.py to perform the final
     classification. If transformation is disabled and only wavelength selection
     is enabled, the Selector class selects a subset of the original wavelengths
     (as prescribed by the chosen wavelength selection algorithm). The selected
     wavelengths are then passed to the MLP class in mlp.py for final
     classification results.

run_selector.py:
  The main point of entry for running the wavelength selection algorithms
  (including the custom genetic algorithm that was developed in this research)
  is the run_selector.py file. This file is where the parameters for the GA
  are set (including crossover rate, mutation rate, etc.). To change the base
  classifier used in the fitness function of the GA, change the 'classifier'
  field in the run_ga_selector() function (for valid options, see the
  _create_classifier() method in genetic_algorithm.py). Results of the genetic
  algorithm are written to a file for further processing by transformer.py

classifiers.py:
  This file is the home for the various classifiers used in this project. All
  classifiers conform to the Classifier abstract base class (ABC), which is
  the equivalent to an interface in other programming languages. All
  implementations of the score() method use 10 fold cross validation to
  return an average score (in terms of accuracy) attained across each fold.

  The classification algorithms include:
    k-nearest neighbors
    decision tree
    logistic regression
    feedforward neural network

  The architecture of the neural network can be customized in the train()
  method of the FFNN class.

utils.py:
  This file contains various utility functions that are not directly related
  to the algorithms implemented. Some are unused in any of the other files,
  as they are remnants from previous directions of research.

genetic_algorithm.py:
  Logic for the genetic algorithm and specific implementations for selection,
  mutation, crossover, and replacement operations.

wavelength_selection.py:
  Home for the Selector astract base class and the corresponding implementations.
  Selectors create an instance of the specified wavelength selection algorithm,
  and run the algorithm to select the wavelengths. These selectors get called
  in the run_selector.py file.

  The find_wavelengths() method of the GASelector class is where the number
  of generations and population size can be set for the genetic algorithm.

hierarchical_clustering.py:
  This is where the logic for the hierarchical clustering algorithm used in the
  HAGRID algorithm is located. There is also functionality for displaying
  the dendrogram produced by hierarchical clustering, and for plotting the
  principal components of the resulting clusters.

mlp.py:
  Contains the multilayer perceptron used for final classification accuracy
  attained using the selected wavelengths. Note that the network here is used
  on the final data (with the selected wavelength subset), where the network
  contained in classifiers.py is used for the fitness function in the GA in
  order to select the wavelengths.

transformer.py:
  The Transformer class transforms the hyperspectral data to look as if it had
  been captured by a multispectral camera. The parameters that impact how this
  is done are passed by final_classify.py:
      transform: boolean - If False, the bandwidths of the hyperspectral data
                           are not set. This means the original hyperspectral
                           bandwidth is retained and no transformation occurs.
                           If True, the hyperspectral data is transformed to
                           have a bandwidth specified by the bandwidth parameter
      bandwidth: boolean - If False, the bandwidth is not determined algorithmically,
                          it is simply set to a default value of 20 nm
                          If True, the histogram of the GA population is used
                          to determine the bandwidth of the transformed data in
                          accordance with the method described in the HAGRID paper

selector.py:
  The Selector class simply selects a subset of wavelengths from the original
  data. This can be the subset determined by the standard GA, the HAGRID method,
  etc. The selected wavelengths are returned and used for classification in
  final_classify.py

final_classify.py:
  Selects wavelengths and transforms data as specified, then runs the new data
  through a neural network to perform the final classification on the selected
  subset of wavelengths. 
