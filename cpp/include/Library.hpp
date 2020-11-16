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

    static BaseModel* CreateModel(int flag, int weights_count);
    static void Train(BaseModel* model, int sample_count, double* train_inputs, int inputs_size, double* train_outputs, int outputs_size, int epochs,
                      double learning_rate);
    static void Predict(BaseModel* model, double* inputs, int inputs_size, double* outputs, int outputs_size);
    static double* GetWeigths(BaseModel* model);
    static void DeleteModel(BaseModel* model);
};

#endif