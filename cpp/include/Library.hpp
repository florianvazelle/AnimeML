#ifndef LIBRARY_HPP
#define LIBRARY_HPP

#include <BaseModel.hpp>

#ifdef WIN32
#    define dllexport __declspec(dllexport)
#else
#    define dllexport
#endif

class Library {
  public:
    enum Flags { LINEAR_MODEL = 0 };

    dllexport BaseModel* CreateModel(int flag, int weights_count);
    dllexport void Train(BaseModel* model, int sample_count, double* train_inputs, int inputs_size, double* train_outputs, int outputs_size,
                        int epochs,
                      double learning_rate);
    dllexport void Predict(BaseModel* model, double* inputs, int inputs_size, double* outputs, int outputs_size);
    dllexport double* GetWeigths(BaseModel* model);
    dllexport void DeleteModel(BaseModel* model);
};

#endif