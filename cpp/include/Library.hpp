#ifndef LIBRARY_HPP
#define LIBRARY_HPP

#ifdef WIN32
#    ifdef example_EXPORTS // <libname>_EXPORTS is a macro added only for shared libraries
#        define DLLEXPORT __declspec(dllexport) // Enabled as "export" while compiling the dll project
#    else
#        define DLLEXPORT __declspec(dllimport) // Enabled as "import" in the Client side for using already created dll file
#    endif
#else
#    define DLLEXPORT
#endif

#include <LinearModel.hpp>
#include <MLP.hpp>
#include <ImageManager.hpp>

/**
 * Expose Library methods.
 *
 * extern "C" specifies that the function is defined
 * elsewhere and uses the C-language calling convention.
 */
extern "C"
{
    DLLEXPORT BaseModel* CreateModel(int flag, int weights_count, bool is_classification);
    DLLEXPORT void Train(BaseModel* model, int sample_count, const double* train_inputs, int inputs_size, const double* train_outputs, int outputs_size, int epochs, double learning_rate);
    DLLEXPORT void Predict(BaseModel* model, int sample_count, const double* inputs, int inputs_size, double* outputs, int outputs_size);
    DLLEXPORT double* GetWeigths(BaseModel* model);
    DLLEXPORT void SaveModel(BaseModel* model, const char* path);
    DLLEXPORT void LoadModel(BaseModel* model, const char* path);
    DLLEXPORT void DeleteModel(BaseModel* model);
    DLLEXPORT void LoadAsset();
};

#endif