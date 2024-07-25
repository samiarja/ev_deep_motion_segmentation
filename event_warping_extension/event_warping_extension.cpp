
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <Python.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <array>
#include <cmath>
#include <limits>
#include <numpy/arrayobject.h>
#include <stdexcept>
#include <vector>

static PyObject* smooth_histogram(PyObject* self, PyObject* args) {
    PyObject* raw_values;
    if (!PyArg_ParseTuple(args, "O", &raw_values)) {
        return nullptr;
    }
    try {
        if (!PyArray_Check(raw_values)) {
            throw std::runtime_error("values must be a numpy array");
        }
        auto values = reinterpret_cast<PyArrayObject*>(raw_values);
        if (PyArray_NDIM(values) != 1) {
            throw std::runtime_error("values's dimension must be 1");
        }
        if (PyArray_TYPE(values) != NPY_FLOAT64) {
            throw std::runtime_error("values's type must be float");
        }
        const auto size = PyArray_SIZE(values);
        auto minimum = std::numeric_limits<double>::infinity();
        auto maximum = -std::numeric_limits<double>::infinity();
        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            const auto value = *reinterpret_cast<double*>(PyArray_GETPTR1(values, index));
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        Py_END_ALLOW_THREADS;
        const npy_intp dimension = static_cast<npy_intp>(std::ceil(maximum - minimum + 1)) + 2;
        auto result = reinterpret_cast<PyArrayObject*>(
            PyArray_Zeros(1, &dimension, PyArray_DescrFromType(NPY_FLOAT64), 0));
        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            const auto value =
                (*reinterpret_cast<double*>(PyArray_GETPTR1(values, index))) - minimum + 1.0;
            const auto value_i = std::floor(value);
            const auto value_f = value - value_i;
            (*reinterpret_cast<double*>(PyArray_GETPTR1(result, static_cast<npy_intp>(value_i)))) +=
                (1.0 - value_f);
            (*reinterpret_cast<double*>(
                PyArray_GETPTR1(result, static_cast<npy_intp>(value_i) + 1))) += value_f;
        }
        Py_END_ALLOW_THREADS;
        return reinterpret_cast<PyObject*>(result);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

// accumulate is a 2D version of smooth_histogram
static PyObject* accumulate(PyObject* self, PyObject* args) {
    int32_t sensor_width;
    int32_t sensor_height;
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    double velocity_x;
    double velocity_y;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOdd",
            &sensor_width,
            &sensor_height,
            &raw_ts,
            &raw_xs,
            &raw_ys,
            &velocity_x,
            &velocity_y)) {
        return nullptr;
    }
    try {
        if (sensor_width <= 0) {
            throw std::runtime_error("sensor_width must be larger than zero");
        }
        if (sensor_height <= 0) {
            throw std::runtime_error("sensor_height must be larger than zero");
        }
        if (!PyArray_Check(raw_ts)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
        if (PyArray_NDIM(ts) != 1) {
            throw std::runtime_error("t's dimension must be 1");
        }
        if (PyArray_TYPE(ts) != NPY_FLOAT64) {
            throw std::runtime_error("t's type must be float");
        }
        if (!PyArray_Check(raw_xs)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
        if (PyArray_NDIM(xs) != 1) {
            throw std::runtime_error("x's dimension must be 1");
        }
        if (PyArray_TYPE(xs) != NPY_FLOAT64) {
            throw std::runtime_error("x's type must be float");
        }
        if (!PyArray_Check(raw_ys)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
        if (PyArray_NDIM(ys) != 1) {
            throw std::runtime_error("y's dimension must be 1");
        }
        if (PyArray_TYPE(ys) != NPY_FLOAT64) {
            throw std::runtime_error("y's type must be float");
        }
        const auto size = PyArray_SIZE(ts);
        if (PyArray_SIZE(xs) != size) {
            throw std::runtime_error("t and x must have the same size");
        }
        if (PyArray_SIZE(ys) != size) {
            throw std::runtime_error("t and y must have the same size");
        }
        const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        const auto maximum_delta_x = std::floor(std::abs(velocity_x) * (t1 - t0)) + 2.0;
        const auto maximum_delta_y = std::floor(std::abs(velocity_y) * (t1 - t0)) + 2.0;
        const auto width = static_cast<std::size_t>(sensor_width + maximum_delta_x);
        const auto height = static_cast<std::size_t>(sensor_height + maximum_delta_y);
        const std::array<npy_intp, 2> dimensions{
            static_cast<npy_intp>(height), static_cast<npy_intp>(width)};
        auto result = reinterpret_cast<PyArrayObject*>(
            PyArray_Zeros(2, dimensions.data(), PyArray_DescrFromType(NPY_FLOAT64), 0));
        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            const auto warped_x = *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index)) - velocity_x * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto warped_y = *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index)) - velocity_y * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);

            const auto x = warped_x + (velocity_x > 0 ? maximum_delta_x : 0.0);
            const auto y = warped_y + (velocity_y > 0 ? maximum_delta_y : 0.0);
            auto xi = std::floor(x);
            auto yi = std::floor(y);
            const auto xf = x - xi;
            const auto yf = y - yi;
            (*reinterpret_cast<double*>(
                PyArray_GETPTR2(result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi)))) +=
                (1.0 - xf) * (1.0 - yf);
            (*reinterpret_cast<double*>(PyArray_GETPTR2(
                result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi) + 1))) +=
                xf * (1.0 - yf);
            (*reinterpret_cast<double*>(PyArray_GETPTR2(
                result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi)))) +=
                (1.0 - xf) * yf;
            (*reinterpret_cast<double*>(PyArray_GETPTR2(
                result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi) + 1))) += xf * yf;
        }
        Py_END_ALLOW_THREADS;
        return reinterpret_cast<PyObject*>(result);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

//accumulate events in a time surface
static PyObject* accumulate_timesurface(PyObject* self, PyObject* args) {
    int32_t sensor_width;
    int32_t sensor_height;
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    double velocity_x;
    double velocity_y;
    double tau;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOdd|d",
            &sensor_width,
            &sensor_height,
            &raw_ts,
            &raw_xs,
            &raw_ys,
            &velocity_x,
            &velocity_y,
            &tau)) {
        return nullptr;
    }
    try {
        if (sensor_width <= 0) {
            throw std::runtime_error("sensor_width must be larger than zero");
        }
        if (sensor_height <= 0) {
            throw std::runtime_error("sensor_height must be larger than zero");
        }
        if (!PyArray_Check(raw_ts)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
        if (PyArray_NDIM(ts) != 1) {
            throw std::runtime_error("t's dimension must be 1");
        }
        if (PyArray_TYPE(ts) != NPY_FLOAT64) {
            throw std::runtime_error("t's type must be float");
        }
        if (!PyArray_Check(raw_xs)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
        if (PyArray_NDIM(xs) != 1) {
            throw std::runtime_error("x's dimension must be 1");
        }
        if (PyArray_TYPE(xs) != NPY_FLOAT64) {
            throw std::runtime_error("x's type must be float");
        }
        if (!PyArray_Check(raw_ys)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
        if (PyArray_NDIM(ys) != 1) {
            throw std::runtime_error("y's dimension must be 1");
        }
        if (PyArray_TYPE(ys) != NPY_FLOAT64) {
            throw std::runtime_error("y's type must be float");
        }
        const auto size = PyArray_SIZE(ts);
        if (PyArray_SIZE(xs) != size) {
            throw std::runtime_error("t and x must have the same size");
        }
        if (PyArray_SIZE(ys) != size) {
            throw std::runtime_error("t and y must have the same size");
        }
        const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        const auto maximum_delta_x = std::floor(std::abs(velocity_x) * (t1 - t0)) + 2.0;
        const auto maximum_delta_y = std::floor(std::abs(velocity_y) * (t1 - t0)) + 2.0;
        const auto width = static_cast<std::size_t>(sensor_width + maximum_delta_x);
        const auto height = static_cast<std::size_t>(sensor_height + maximum_delta_y);
        const std::array<npy_intp, 2> dimensions{
            static_cast<npy_intp>(height), static_cast<npy_intp>(width)};
        auto result = reinterpret_cast<PyArrayObject*>(
            PyArray_Zeros(2, dimensions.data(), PyArray_DescrFromType(NPY_FLOAT64), 0));
        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            const auto warped_x = *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index)) - velocity_x * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto warped_y = *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index)) - velocity_y * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);

            const auto x = warped_x + (velocity_x > 0 ? maximum_delta_x : 0.0);
            const auto y = warped_y + (velocity_y > 0 ? maximum_delta_y : 0.0);
            auto xi = static_cast<npy_intp>(std::floor(x));
            auto yi = static_cast<npy_intp>(std::floor(y));
            
            if (xi >= 0 && static_cast<std::size_t>(xi) < width && yi >= 0 && static_cast<std::size_t>(yi) < height) {
                double* current_pixel_value = reinterpret_cast<double*>(PyArray_GETPTR2(result, yi, xi));
                double t_current = *current_pixel_value;

                // Apply decay equation
                double S = std::exp((t_current - (*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index)))) / tau);
                
                // Update the pixel with the new timestamp
                *current_pixel_value = (*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) * S;
            }
        }
        Py_END_ALLOW_THREADS;
        return reinterpret_cast<PyObject*>(result);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

// make accumulate return another array about the new pixel for each events after warping
static PyObject* accumulate_pixel_map(PyObject* self, PyObject* args) {
    int32_t sensor_width;
    int32_t sensor_height;
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    double velocity_x;
    double velocity_y;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOdd",
            &sensor_width,
            &sensor_height,
            &raw_ts,
            &raw_xs,
            &raw_ys,
            &velocity_x,
            &velocity_y)) {
        return nullptr;
    }
    try {
        if (sensor_width <= 0) {
            throw std::runtime_error("sensor_width must be larger than zero");
        }
        if (sensor_height <= 0) {
            throw std::runtime_error("sensor_height must be larger than zero");
        }
        if (!PyArray_Check(raw_ts)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
        if (PyArray_NDIM(ts) != 1) {
            throw std::runtime_error("t's dimension must be 1");
        }
        if (PyArray_TYPE(ts) != NPY_FLOAT64) {
            throw std::runtime_error("t's type must be float");
        }
        if (!PyArray_Check(raw_xs)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
        if (PyArray_NDIM(xs) != 1) {
            throw std::runtime_error("x's dimension must be 1");
        }
        if (PyArray_TYPE(xs) != NPY_FLOAT64) {
            throw std::runtime_error("x's type must be float");
        }
        if (!PyArray_Check(raw_ys)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
        if (PyArray_NDIM(ys) != 1) {
            throw std::runtime_error("y's dimension must be 1");
        }
        if (PyArray_TYPE(ys) != NPY_FLOAT64) {
            throw std::runtime_error("y's type must be float");
        }
        const auto size = PyArray_SIZE(ts);
        if (PyArray_SIZE(xs) != size) {
            throw std::runtime_error("t and x must have the same size");
        }
        if (PyArray_SIZE(ys) != size) {
            throw std::runtime_error("t and y must have the same size");
        }
        const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        const auto maximum_delta_x = std::floor(std::abs(velocity_x) * (t1 - t0)) + 2.0;
        const auto maximum_delta_y = std::floor(std::abs(velocity_y) * (t1 - t0)) + 2.0;
        const auto width = static_cast<std::size_t>(sensor_width + maximum_delta_x);
        const auto height = static_cast<std::size_t>(sensor_height + maximum_delta_y);
        const std::array<npy_intp, 2> dimensions{
            static_cast<npy_intp>(height), static_cast<npy_intp>(width)};
        auto result = reinterpret_cast<PyArrayObject*>(
            PyArray_Zeros(2, dimensions.data(), PyArray_DescrFromType(NPY_FLOAT64), 0));
        std::string error_message;
        bool has_error = false;

        // Define a structure to hold the event indices temporarily
        std::vector<std::vector<std::vector<npy_intp>>> event_indices(height, std::vector<std::vector<npy_intp>>(width));

        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            auto t_ptr = reinterpret_cast<double*>(PyArray_GETPTR1(ts, index));
            if (!t_ptr) {
                // Store error state and message
                has_error = true;
                error_message = "Failed to get pointer to t array";
                break; // Exit the loop
            }
            double warped_x = *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index)) - velocity_x * (*t_ptr - t0);
            double warped_y = *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index)) - velocity_y * (*t_ptr - t0);

            double x = std::max(0.0, std::min(warped_x + (velocity_x > 0 ? maximum_delta_x : 0.0), static_cast<double>(width - 2)));
            double y = std::max(0.0, std::min(warped_y + (velocity_y > 0 ? maximum_delta_y : 0.0), static_cast<double>(height - 2)));

            auto xi = static_cast<npy_intp>(std::floor(x));
            auto yi = static_cast<npy_intp>(std::floor(y));

            // Add checks to ensure xi and yi are within the bounds
            if (xi < 0 || xi >= width - 1 || yi < 0 || yi >= height - 1) {
                continue; // Skip the event if it is out of bounds
            }

            const auto xf = x - xi;
            const auto yf = y - yi;

            // Use safe pointer access for updating the result array
            double* ptr00 = reinterpret_cast<double*>(PyArray_GETPTR2(result, yi, xi));
            double* ptr01 = reinterpret_cast<double*>(PyArray_GETPTR2(result, yi, xi + 1));
            double* ptr10 = reinterpret_cast<double*>(PyArray_GETPTR2(result, yi + 1, xi));
            double* ptr11 = reinterpret_cast<double*>(PyArray_GETPTR2(result, yi + 1, xi + 1));

            if (!ptr00 || !ptr01 || !ptr10 || !ptr11) {
                // If any pointer is null, exit the loop and set an error
                error_message = "Failed to get pointer to result array";
                has_error = true;
                break;
            }

            // Now it's safe to update the values
            *ptr00 += (1.0 - xf) * (1.0 - yf);
            *ptr01 += xf * (1.0 - yf);
            *ptr10 += (1.0 - xf) * yf;
            *ptr11 += xf * yf;

            // Update event_indices ensuring we are within bounds
            if (yi < height && xi < width) {
                event_indices[yi][xi].push_back(index);
                if (xi + 1 < width) event_indices[yi][xi + 1].push_back(index);
                if (yi + 1 < height) event_indices[yi + 1][xi].push_back(index);
                if (xi + 1 < width && yi + 1 < height) event_indices[yi + 1][xi + 1].push_back(index);
            }
        }
        Py_END_ALLOW_THREADS;

        // After the thread block, check for errors and handle them
        if (has_error) {
            PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
            return nullptr;
        }

        // Acquire GIL here before working with Python objects
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        PyObject* listObj = PyList_New(0);
        for (const auto& row : event_indices) {
            PyObject* rowList = PyList_New(0);
            for (const auto& col : row) {
                PyObject* colList = PyList_New(0);
                for (const auto& idx : col) {
                    PyObject* indexObj = PyLong_FromLong(idx);
                    PyList_Append(colList, indexObj);
                    Py_DECREF(indexObj); // Decrement the reference count immediately after appending
                }
                PyList_Append(rowList, colList);
                Py_DECREF(colList); // Decrement reference count of the column list
            }
            PyList_Append(listObj, rowList);
            Py_DECREF(rowList); // Decrement reference count of the row list
        }

        PyGILState_Release(gstate); // Release GIL after working with Python objects

        // Return the result and the event indices list
        return Py_BuildValue("OO", result, listObj);

    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}


// call this function if vx and vy are both numpy array
static PyObject* accumulate_cnt(PyObject* self, PyObject* args) {
    int32_t sensor_width;
    int32_t sensor_height;
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    PyObject* raw_velocity_x;
    PyObject* raw_velocity_y; 
    if (!PyArg_ParseTuple(
            args,
            "iiOOOOO",
            &sensor_width,
            &sensor_height,
            &raw_ts,
            &raw_xs,
            &raw_ys,
            &raw_velocity_x,
            &raw_velocity_y)) {
        return nullptr;
    }
    try {
        if (sensor_width <= 0) {
            throw std::runtime_error("sensor_width must be larger than zero");
        }
        if (sensor_height <= 0) {
            throw std::runtime_error("sensor_height must be larger than zero");
        }
        if (!PyArray_Check(raw_ts)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
        if (PyArray_NDIM(ts) != 1) {
            throw std::runtime_error("t's dimension must be 1");
        }
        if (PyArray_TYPE(ts) != NPY_FLOAT64) {
            throw std::runtime_error("t's type must be float");
        }
        if (!PyArray_Check(raw_xs)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
        if (PyArray_NDIM(xs) != 1) {
            throw std::runtime_error("x's dimension must be 1");
        }
        if (PyArray_TYPE(xs) != NPY_FLOAT64) {
            throw std::runtime_error("x's type must be float");
        }
        if (!PyArray_Check(raw_ys)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
        if (PyArray_NDIM(ys) != 1) {
            throw std::runtime_error("y's dimension must be 1");
        }
        if (PyArray_TYPE(ys) != NPY_FLOAT64) {
            throw std::runtime_error("y's type must be float");
        }
        const auto size = PyArray_SIZE(ts);
        if (PyArray_SIZE(xs) != size) {
            throw std::runtime_error("t and x must have the same size");
        }
        if (PyArray_SIZE(ys) != size) {
            throw std::runtime_error("t and y must have the same size");
        }
        auto velocity_x = reinterpret_cast<PyArrayObject*>(raw_velocity_x); 
        auto velocity_y = reinterpret_cast<PyArrayObject*>(raw_velocity_y);
        const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        double *c_velocity_x = reinterpret_cast<double*>(PyArray_DATA(velocity_x));
        double *c_velocity_y = reinterpret_cast<double*>(PyArray_DATA(velocity_y));
        double max_velocity_x = *std::max_element(c_velocity_x, c_velocity_x + size);
        double max_velocity_y = *std::max_element(c_velocity_y, c_velocity_y + size);

        const auto maximum_delta_x = std::floor(std::abs(max_velocity_x) * (t1 - t0)) + 2.0;
        const auto maximum_delta_y = std::floor(std::abs(max_velocity_y) * (t1 - t0)) + 2.0;
        const auto width = static_cast<std::size_t>(sensor_width + maximum_delta_x);
        const auto height = static_cast<std::size_t>(sensor_height + maximum_delta_y);
        const std::array<npy_intp, 2> dimensions{
            static_cast<npy_intp>(height), static_cast<npy_intp>(width)};
        auto result = reinterpret_cast<PyArrayObject*>(
            PyArray_Zeros(2, dimensions.data(), PyArray_DescrFromType(NPY_FLOAT64), 0));
        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            const auto warped_x = *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index)) - (*reinterpret_cast<double*>(PyArray_GETPTR1(velocity_x, index))) * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto warped_y = *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index)) - (*reinterpret_cast<double*>(PyArray_GETPTR1(velocity_y, index))) * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);

            const auto vx_i = *reinterpret_cast<double*>(PyArray_GETPTR1(velocity_x, index));
            const auto vy_i = *reinterpret_cast<double*>(PyArray_GETPTR1(velocity_y, index));

            const auto x = warped_x + (vx_i > 0 ? maximum_delta_x : 0.0);
            const auto y = warped_y + (vy_i > 0 ? maximum_delta_y : 0.0);

            auto xi = std::floor(x);
            auto yi = std::floor(y);
            const auto xf = x - xi;
            const auto yf = y - yi;
            (*reinterpret_cast<double*>(
                PyArray_GETPTR2(result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi)))) +=
                (1.0 - xf) * (1.0 - yf);
            (*reinterpret_cast<double*>(PyArray_GETPTR2(
                result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi) + 1))) +=
                xf * (1.0 - yf);
            (*reinterpret_cast<double*>(PyArray_GETPTR2(
                result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi)))) +=
                (1.0 - xf) * yf;
            (*reinterpret_cast<double*>(PyArray_GETPTR2(
                result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi) + 1))) += xf * yf;
        }
        Py_END_ALLOW_THREADS;
        return reinterpret_cast<PyObject*>(result);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

//this is the same as accumulate_cnt but for motion segmentation
static PyObject* accumulate_cnt_rgb(PyObject* self, PyObject* args) {
    int32_t sensor_width;
    int32_t sensor_height;
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    PyObject* raw_labels; 
    PyObject* raw_velocity_x;
    PyObject* raw_velocity_y; 
    if (!PyArg_ParseTuple(
            args,
            "iiOOOOOO",
            &sensor_width,
            &sensor_height,
            &raw_ts,
            &raw_xs,
            &raw_ys,
            &raw_labels,
            &raw_velocity_x,
            &raw_velocity_y)) {
        return nullptr;
    }
    try {
        if (sensor_width <= 0) {
            throw std::runtime_error("sensor_width must be larger than zero");
        }
        if (sensor_height <= 0) {
            throw std::runtime_error("sensor_height must be larger than zero");
        }
        if (!PyArray_Check(raw_ts)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
        if (PyArray_NDIM(ts) != 1) {
            throw std::runtime_error("t's dimension must be 1");
        }
        if (PyArray_TYPE(ts) != NPY_FLOAT64) {
            throw std::runtime_error("t's type must be float");
        }
        if (!PyArray_Check(raw_xs)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
        if (PyArray_NDIM(xs) != 1) {
            throw std::runtime_error("x's dimension must be 1");
        }
        if (PyArray_TYPE(xs) != NPY_FLOAT64) {
            throw std::runtime_error("x's type must be float");
        }
        if (!PyArray_Check(raw_ys)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
        if (PyArray_NDIM(ys) != 1) {
            throw std::runtime_error("y's dimension must be 1");
        }
        if (PyArray_TYPE(ys) != NPY_FLOAT64) {
            throw std::runtime_error("y's type must be float");
        }
        const auto size = PyArray_SIZE(ts);
        if (PyArray_SIZE(xs) != size) {
            throw std::runtime_error("t and x must have the same size");
        }
        if (PyArray_SIZE(ys) != size) {
            throw std::runtime_error("t and y must have the same size");
        }
        // Check and extract raw_labels
        if (!PyArray_Check(raw_labels)) {
            throw std::runtime_error("labels must be a numpy array");
        }
        auto labels = reinterpret_cast<PyArrayObject*>(raw_labels);
        if (PyArray_NDIM(labels) != 1) {
            throw std::runtime_error("labels's dimension must be 1");
        }
        if (PyArray_TYPE(labels) != NPY_INT32) {
            throw std::runtime_error("labels's type must be int32");
        }

        auto velocity_x = reinterpret_cast<PyArrayObject*>(raw_velocity_x); 
        auto velocity_y = reinterpret_cast<PyArrayObject*>(raw_velocity_y);
        const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        double *c_velocity_x = reinterpret_cast<double*>(PyArray_DATA(velocity_x));
        double *c_velocity_y = reinterpret_cast<double*>(PyArray_DATA(velocity_y));
        double max_velocity_x = *std::max_element(c_velocity_x, c_velocity_x + size);
        double max_velocity_y = *std::max_element(c_velocity_y, c_velocity_y + size);

        const auto maximum_delta_x = std::floor(std::abs(max_velocity_x) * (t1 - t0)) + 2.0;
        const auto maximum_delta_y = std::floor(std::abs(max_velocity_y) * (t1 - t0)) + 2.0;
        const auto width = static_cast<std::size_t>(sensor_width + maximum_delta_x);
        const auto height = static_cast<std::size_t>(sensor_height + maximum_delta_y);
        const std::array<npy_intp, 2> dimensions{
            static_cast<npy_intp>(height), static_cast<npy_intp>(width)};
        auto result = reinterpret_cast<PyArrayObject*>(
            PyArray_Zeros(2, dimensions.data(), PyArray_DescrFromType(NPY_FLOAT64), 0));
        auto result_labels = reinterpret_cast<PyArrayObject*>(
            PyArray_Zeros(2, dimensions.data(), PyArray_DescrFromType(NPY_INT32), 0));
        Py_BEGIN_ALLOW_THREADS;
        for (npy_intp index = 0; index < size; ++index) {
            const auto warped_x = *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index)) - (*reinterpret_cast<double*>(PyArray_GETPTR1(velocity_x, index))) * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto warped_y = *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index)) - (*reinterpret_cast<double*>(PyArray_GETPTR1(velocity_y, index))) * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);

            const auto vx_i = *reinterpret_cast<double*>(PyArray_GETPTR1(velocity_x, index));
            const auto vy_i = *reinterpret_cast<double*>(PyArray_GETPTR1(velocity_y, index));

            const auto x = warped_x + (vx_i > 0 ? maximum_delta_x : 0.0);
            const auto y = warped_y + (vy_i > 0 ? maximum_delta_y : 0.0);

            auto xi = std::floor(x);
            auto yi = std::floor(y);
            const auto xf = x - xi;
            const auto yf = y - yi;

            if (yi >= 0 && yi < height - 1 && xi >= 0 && xi < width - 1) {
                (*reinterpret_cast<double*>(
                    PyArray_GETPTR2(result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi)))) +=
                    (1.0 - xf) * (1.0 - yf);
                (*reinterpret_cast<double*>(PyArray_GETPTR2(
                    result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi) + 1))) +=
                    xf * (1.0 - yf);
                (*reinterpret_cast<double*>(PyArray_GETPTR2(
                    result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi)))) +=
                    (1.0 - xf) * yf;
                (*reinterpret_cast<double*>(PyArray_GETPTR2(
                    result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi) + 1))) += xf * yf;

                const auto event_label = *reinterpret_cast<int*>(PyArray_GETPTR1(labels, index));
                *reinterpret_cast<int*>(PyArray_GETPTR2(result_labels, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi))) = event_label;
                *reinterpret_cast<int*>(PyArray_GETPTR2(result_labels, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi) + 1)) = event_label;
                *reinterpret_cast<int*>(PyArray_GETPTR2(result_labels, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi))) = event_label;
                *reinterpret_cast<int*>(PyArray_GETPTR2(result_labels, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi) + 1)) = event_label;
            }
        }
        Py_END_ALLOW_THREADS;
        return Py_BuildValue("OO", result, result_labels);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

// static PyObject* geometricTransformation(PyObject* self, PyObject* args) {
//     int32_t resolution;
//     double rotation_angle;

//     if (!PyArg_ParseTuple(args, "id", &resolution, &rotation_angle)) {
//         return nullptr;
//     }

//     double angle = rotation_angle * 2 * M_PI / 100.0;  // convert to radians
//     double size = 1;

//     std::vector<double> particles_x(resolution);
//     std::vector<double> particles_y(resolution);
//     std::vector<double> particles_z(resolution);
//     std::vector<Eigen::Vector3d> particles(resolution * resolution * resolution);

//     double step = 2.0 * size / (resolution - 1);
//     for (int32_t i = 0; i < resolution; i++) {
//         particles_x[i] = particles_y[i] = particles_z[i] = -size + i * step;
//     }

//     int32_t idx = 0;
//     for (const auto& x : particles_x) {
//         for (const auto& y : particles_y) {
//             for (const auto& z : particles_z) {
//                 particles[idx++] = Eigen::Vector3d(x, y, z);
//             }
//         }
//     }

//     std::vector<Eigen::Vector3d> rotated_particles(resolution * resolution * resolution);
//     for (size_t i = 0; i < particles.size(); i++) {
//         double theta = (particles[i][2] - particles_z.front()) * angle;
//         Eigen::Matrix3d rotation_matrix;
//         rotation_matrix << std::cos(theta), -std::sin(theta), 0,
//                            std::sin(theta),  std::cos(theta), 0,
//                            0,                0,               1;
//         rotated_particles[i] = rotation_matrix * particles[i];
//     }

//     PyObject* pyList = PyList_New(rotated_particles.size());
//     for (size_t i = 0; i < rotated_particles.size(); i++) {
//         PyObject* pySubList = PyList_New(3);
//         for (int j = 0; j < 3; j++) {
//             PyList_SetItem(pySubList, j, PyFloat_FromDouble(rotated_particles[i][j]));
//         }
//         PyList_SetItem(pyList, i, pySubList);
//     }

//     return pyList;
// }


// // accumulate4D: enable the accumulation process to account for rotation and scaling
// static PyObject* accumulate4D(PyObject* self, PyObject* args) {
//     int32_t sensor_width;
//     int32_t sensor_height;
//     PyObject* raw_ts;
//     PyObject* raw_xs;
//     PyObject* raw_ys;
//     double velocity_x;
//     double velocity_y;
//     double wx;
//     double wy;
//     double wz;
//     double zoom;
//     if (!PyArg_ParseTuple(
//             args,
//             "iiOOOdddddd",
//             &sensor_width,
//             &sensor_height,
//             &raw_ts,
//             &raw_xs,
//             &raw_ys,
//             &velocity_x,
//             &velocity_y,
//             &wx,
//             &wy,
//             &wz,
//             &zoom)) {
//         return nullptr;
//     }
//     try {
//         if (sensor_width <= 0) {
//             throw std::runtime_error("sensor_width must be larger than zero");
//         }
//         if (sensor_height <= 0) {
//             throw std::runtime_error("sensor_height must be larger than zero");
//         }
//         if (!PyArray_Check(raw_ts)) {
//             throw std::runtime_error("t must be a numpy array");
//         }
//         auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
//         if (PyArray_NDIM(ts) != 1) {
//             throw std::runtime_error("t's dimension must be 1");
//         }
//         if (PyArray_TYPE(ts) != NPY_FLOAT64) {
//             throw std::runtime_error("t's type must be float");
//         }
//         if (!PyArray_Check(raw_xs)) {
//             throw std::runtime_error("t must be a numpy array");
//         }
//         auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
//         if (PyArray_NDIM(xs) != 1) {
//             throw std::runtime_error("x's dimension must be 1");
//         }
//         if (PyArray_TYPE(xs) != NPY_FLOAT64) {
//             throw std::runtime_error("x's type must be float");
//         }
//         if (!PyArray_Check(raw_ys)) {
//             throw std::runtime_error("x must be a numpy array");
//         }
//         auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
//         if (PyArray_NDIM(ys) != 1) {
//             throw std::runtime_error("y's dimension must be 1");
//         }
//         if (PyArray_TYPE(ys) != NPY_FLOAT64) {
//             throw std::runtime_error("y's type must be float");
//         }
//         const auto size = PyArray_SIZE(ts);
//         if (PyArray_SIZE(xs) != size) {
//             throw std::runtime_error("t and x must have the same size");
//         }
//         if (PyArray_SIZE(ys) != size) {
//             throw std::runtime_error("t and y must have the same size");
//         }
        
//         Eigen::Matrix3f rot_mat;
//         rot_mat << 0, -wz, wy, 
//                 wz, 0, -wx, 
//                 -wy, wx, 0;

//         const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
//         const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        
//         // Calculate displacement due to rotation
//         double diagonal = std::sqrt(sensor_width * sensor_width + sensor_height * sensor_height);
//         double max_rot_displacement_x = std::max(std::abs(diagonal/2 * std::cos(wz) - diagonal/2 * std::sin(wz)),
//                                                 std::abs(diagonal/2 * std::cos(wz) + diagonal/2 * std::sin(wz)));
//         double min_rot_displacement_x = std::min(std::abs(diagonal/2 * std::cos(wz) - diagonal/2 * std::sin(wz)),
//                                                 std::abs(diagonal/2 * std::cos(wz) + diagonal/2 * std::sin(wz)));
//         double max_rot_displacement_y = std::max(std::abs(diagonal/2 * std::sin(wz) + diagonal/2 * std::cos(wz)),
//                                                 std::abs(-diagonal/2 * std::sin(wz) + diagonal/2 * std::cos(wz)));
//         double min_rot_displacement_y = std::min(std::abs(diagonal/2 * std::sin(wz) + diagonal/2 * std::cos(wz)),
//                                                 std::abs(-diagonal/2 * std::sin(wz) + diagonal/2 * std::cos(wz)));

//         // Calculate displacement due to scaling
//         double max_scale_displacement_x = std::max(0.0, (zoom - 1) * sensor_width / 2);
//         double min_scale_displacement_x = std::min(0.0, (zoom - 1) * sensor_width / 2);
//         double max_scale_displacement_y = std::max(0.0, (zoom - 1) * sensor_height / 2);
//         double min_scale_displacement_y = std::min(0.0, (zoom - 1) * sensor_height / 2);

//         // Calculate displacement due to translation
//         double max_trans_displacement_x = std::max(0.0, velocity_x * (t1 - t0));
//         double min_trans_displacement_x = std::min(0.0, velocity_x * (t1 - t0));
//         double max_trans_displacement_y = std::max(0.0, velocity_y * (t1 - t0));
//         double min_trans_displacement_y = std::min(0.0, velocity_y * (t1 - t0));

//         // Take maximum and minimum of all displacements
//         double max_displacement_x = std::max({
//             max_rot_displacement_x + max_scale_displacement_x + max_trans_displacement_x,
//             -min_rot_displacement_x + min_scale_displacement_x + min_trans_displacement_x
//         });
//         double max_displacement_y = std::max({
//             max_rot_displacement_y + max_scale_displacement_y + max_trans_displacement_y,
//             -min_rot_displacement_y + min_scale_displacement_y + min_trans_displacement_y
//         });

//         double min_displacement_x = std::min({
//             max_rot_displacement_x + max_scale_displacement_x + max_trans_displacement_x,
//             -min_rot_displacement_x + min_scale_displacement_x + min_trans_displacement_x
//         });
//         double min_displacement_y = std::min({
//             max_rot_displacement_y + max_scale_displacement_y + max_trans_displacement_y,
//             -min_rot_displacement_y + min_scale_displacement_y + min_trans_displacement_y
//         });

//         const auto width = static_cast<std::size_t>(sensor_width + max_displacement_x - min_displacement_x);
//         const auto height = static_cast<std::size_t>(sensor_height + max_displacement_y - min_displacement_y);

//         const std::array<npy_intp, 2> dimensions{
//             static_cast<npy_intp>(height), static_cast<npy_intp>(width)};
//         auto result = reinterpret_cast<PyArrayObject*>(
//             PyArray_Zeros(2, dimensions.data(), PyArray_DescrFromType(NPY_FLOAT64), 0));
//         Py_BEGIN_ALLOW_THREADS;
//         for (npy_intp index = 0; index < size; ++index) {
//             const auto t = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, index));
//             const auto dt = t - t0;

//             const auto x_uncentered = *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index));
//             const auto y_uncentered = *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index));

//             // Center the event points around the middle of the sensor
//             const auto x_centered = x_uncentered - sensor_width / 2.0;
//             const auto y_centered = y_uncentered - sensor_height / 2.0;

//             Eigen::Vector3f event_point(x_centered, y_centered, 1.0);
//             Eigen::Matrix3f transform = Eigen::Matrix3f::Identity();

//             // Apply scaling first
//             Eigen::Matrix3f scale_mat = Eigen::Matrix3f::Identity();
//             scale_mat(0, 0) = 1 - dt * zoom;
//             scale_mat(1, 1) = 1 - dt * zoom;
//             transform *= scale_mat;

//             // Then apply the rotation
//             Eigen::Matrix3f rot_exp = rot_mat * dt;
//             rot_exp = rot_exp.exp();
//             transform *= rot_exp;

//             // Finally, apply the desired translation
//             transform(0, 2) += dt * velocity_x;
//             transform(1, 2) += dt * velocity_y;

//             // Translate back to the center of the image
//             transform(0, 2) += sensor_width / 2.0 - min_displacement_x;
//             transform(1, 2) += sensor_height / 2.0 - min_displacement_y;

//             // Apply the transformation matrix to the event point
//             Eigen::Vector3f transformed_point = transform * event_point;
//             const auto x = std::abs(transformed_point[0]);
//             const auto y = std::abs(transformed_point[1]);

//             auto xi = std::floor(x);
//             auto yi = std::floor(y);
//             const auto xf = (x - xi);
//             const auto yf = (y - yi);

//             if ((xi >= 0) && (yi >= 0) && (xi + 1 < width) && (yi + 1 < height)) {
//                 (*reinterpret_cast<double*>(PyArray_GETPTR2(result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi)))) +=
//                     (1.0 - xf) * (1.0 - yf);
//                 (*reinterpret_cast<double*>(PyArray_GETPTR2(result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi) + 1))) +=
//                     xf * (1.0 - yf);
//                 (*reinterpret_cast<double*>(PyArray_GETPTR2(result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi)))) +=
//                     (1.0 - xf) * yf;
//                 (*reinterpret_cast<double*>(PyArray_GETPTR2(result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi) + 1))) +=
//                     xf * yf;
//             }

//         }
//         Py_END_ALLOW_THREADS;
//         return reinterpret_cast<PyObject*>(result);
//     } catch (const std::exception& exception) {
//         PyErr_SetString(PyExc_RuntimeError, exception.what());
//     }
//     return nullptr;
// }

// // use this if translation, rotation and scaling inputs are all numpy array
// static PyObject* accumulate4D_cnt(PyObject* self, PyObject* args) {
//     int32_t sensor_width;
//     int32_t sensor_height;
//     PyObject* raw_ts;
//     PyObject* raw_xs;
//     PyObject* raw_ys;
//     PyObject* raw_velocity_x;
//     PyObject* raw_velocity_y;
//     PyObject* raw_wx;
//     PyObject* raw_wy;
//     PyObject* raw_wz;
//     PyObject* raw_zoom;
//     if (!PyArg_ParseTuple(
//             args,
//             "iiOOOOOOOOO",
//             &sensor_width,
//             &sensor_height,
//             &raw_ts,
//             &raw_xs,
//             &raw_ys,
//             &raw_velocity_x,
//             &raw_velocity_y,
//             &raw_wx,
//             &raw_wy,
//             &raw_wz,
//             &raw_zoom)) {
//         return nullptr;
//     }
//     try {
//         if (sensor_width <= 0) {
//             throw std::runtime_error("sensor_width must be larger than zero");
//         }
//         if (sensor_height <= 0) {
//             throw std::runtime_error("sensor_height must be larger than zero");
//         }
//         if (!PyArray_Check(raw_ts)) {
//             throw std::runtime_error("t must be a numpy array");
//         }
//         auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
//         if (PyArray_NDIM(ts) != 1) {
//             throw std::runtime_error("t's dimension must be 1");
//         }
//         if (PyArray_TYPE(ts) != NPY_FLOAT64) {
//             throw std::runtime_error("t's type must be float");
//         }
//         if (!PyArray_Check(raw_xs)) {
//             throw std::runtime_error("t must be a numpy array");
//         }
//         auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
//         if (PyArray_NDIM(xs) != 1) {
//             throw std::runtime_error("x's dimension must be 1");
//         }
//         if (PyArray_TYPE(xs) != NPY_FLOAT64) {
//             throw std::runtime_error("x's type must be float");
//         }
//         if (!PyArray_Check(raw_ys)) {
//             throw std::runtime_error("x must be a numpy array");
//         }
//         auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
//         if (PyArray_NDIM(ys) != 1) {
//             throw std::runtime_error("y's dimension must be 1");
//         }
//         if (PyArray_TYPE(ys) != NPY_FLOAT64) {
//             throw std::runtime_error("y's type must be float");
//         }
//         const auto size = PyArray_SIZE(ts);
//         if (PyArray_SIZE(xs) != size) {
//             throw std::runtime_error("t and x must have the same size");
//         }
//         if (PyArray_SIZE(ys) != size) {
//             throw std::runtime_error("t and y must have the same size");
//         }
//         if (!PyArray_Check(raw_velocity_x)) {
//         throw std::runtime_error("velocity_x must be a numpy array");
//         }
//         auto velocity_x = reinterpret_cast<PyArrayObject*>(raw_velocity_x);
//         if (PyArray_NDIM(velocity_x) != 1) {
//             throw std::runtime_error("velocity_x's dimension must be 1");
//         }
//         if (PyArray_TYPE(velocity_x) != NPY_FLOAT64) {
//             throw std::runtime_error("velocity_x's type must be float");
//         }
//         if (!PyArray_Check(raw_velocity_y)) {
//             throw std::runtime_error("velocity_y must be a numpy array");
//         }
//         auto velocity_y = reinterpret_cast<PyArrayObject*>(raw_velocity_y);
//         if (PyArray_NDIM(velocity_y) != 1) {
//             throw std::runtime_error("velocity_y's dimension must be 1");
//         }
//         if (PyArray_TYPE(velocity_y) != NPY_FLOAT64) {
//             throw std::runtime_error("velocity_y's type must be float");
//         }

//         if (!PyArray_Check(raw_wx)) {
//             throw std::runtime_error("wx must be a numpy array");
//         }
//         auto wx = reinterpret_cast<PyArrayObject*>(raw_wx);
//         if (PyArray_NDIM(wx) != 1) {
//             throw std::runtime_error("wx's dimension must be 1");
//         }
//         if (PyArray_TYPE(wx) != NPY_FLOAT64) {
//             throw std::runtime_error("wx's type must be float");
//         }

//         if (!PyArray_Check(raw_wy)) {
//             throw std::runtime_error("wy must be a numpy array");
//         }
//         auto wy = reinterpret_cast<PyArrayObject*>(raw_wy);
//         if (PyArray_NDIM(wy) != 1) {
//             throw std::runtime_error("wy's dimension must be 1");
//         }
//         if (PyArray_TYPE(wy) != NPY_FLOAT64) {
//             throw std::runtime_error("wy's type must be float");
//         }

//         if (!PyArray_Check(raw_wz)) {
//             throw std::runtime_error("wz must be a numpy array");
//         }
//         auto wz = reinterpret_cast<PyArrayObject*>(raw_wz);
//         if (PyArray_NDIM(wz) != 1) {
//             throw std::runtime_error("wz's dimension must be 1");
//         }
//         if (PyArray_TYPE(wz) != NPY_FLOAT64) {
//             throw std::runtime_error("wz's type must be float");
//         }

//         if (!PyArray_Check(raw_zoom)) {
//             throw std::runtime_error("zoom must be a numpy array");
//         }
//         auto zoom = reinterpret_cast<PyArrayObject*>(raw_zoom);
//         if (PyArray_NDIM(zoom) != 1) {
//             throw std::runtime_error("zoom's dimension must be 1");
//         }
//         if (PyArray_TYPE(zoom) != NPY_FLOAT64) {
//             throw std::runtime_error("zoom's type must be float");
//         }
        
//         const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        
//         // Calculate displacement due to rotation
//         double diagonal = std::sqrt(sensor_width * sensor_width + sensor_height * sensor_height);
        
//         std::vector<double> max_rot_displacement_x(size);
//         std::vector<double> min_rot_displacement_x(size);
//         std::vector<double> max_rot_displacement_y(size);
//         std::vector<double> min_rot_displacement_y(size);
//         std::vector<double> max_scale_displacement_x(size);
//         std::vector<double> min_scale_displacement_x(size);
//         std::vector<double> max_scale_displacement_y(size);
//         std::vector<double> min_scale_displacement_y(size);
//         std::vector<double> max_trans_displacement_x(size);
//         std::vector<double> min_trans_displacement_x(size);
//         std::vector<double> max_trans_displacement_y(size);
//         std::vector<double> min_trans_displacement_y(size);
        
//         for (npy_intp index = 0; index < size; ++index) {
//             const auto t = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, index));
//             const auto dt = t - t0;
//             const auto velocity_x_i = *reinterpret_cast<double*>(PyArray_GETPTR1(velocity_x, index));
//             const auto velocity_y_i = *reinterpret_cast<double*>(PyArray_GETPTR1(velocity_y, index));
//             const auto wx_i = *reinterpret_cast<double*>(PyArray_GETPTR1(wx, index));
//             const auto wy_i = *reinterpret_cast<double*>(PyArray_GETPTR1(wy, index));
//             const auto wz_i = *reinterpret_cast<double*>(PyArray_GETPTR1(wz, index));
//             const auto zoom_i = *reinterpret_cast<double*>(PyArray_GETPTR1(zoom, index));
            
//             max_rot_displacement_x[index] = std::max(std::abs(diagonal/2 * std::cos(wz_i) - diagonal/2 * std::sin(wz_i)),
//                                                     std::abs(diagonal/2 * std::cos(wz_i) + diagonal/2 * std::sin(wz_i)));
//             min_rot_displacement_x[index] = std::min(std::abs(diagonal/2 * std::cos(wz_i) - diagonal/2 * std::sin(wz_i)),
//                                                     std::abs(diagonal/2 * std::cos(wz_i) + diagonal/2 * std::sin(wz_i)));
//             max_rot_displacement_y[index] = std::max(std::abs(diagonal/2 * std::sin(wz_i) + diagonal/2 * std::cos(wz_i)),
//                                                     std::abs(-diagonal/2 * std::sin(wz_i) + diagonal/2 * std::cos(wz_i)));
//             min_rot_displacement_y[index] = std::min(std::abs(diagonal/2 * std::sin(wz_i) + diagonal/2 * std::cos(wz_i)),
//                                                     std::abs(-diagonal/2 * std::sin(wz_i) + diagonal/2 * std::cos(wz_i)));
//             max_scale_displacement_x[index] = std::max(0.0, (zoom_i - 1) * sensor_width / 2);
//             min_scale_displacement_x[index] = std::min(0.0, (zoom_i - 1) * sensor_width / 2);
//             max_scale_displacement_y[index] = std::max(0.0, (zoom_i - 1) * sensor_height / 2);
//             min_scale_displacement_y[index] = std::min(0.0, (zoom_i - 1) * sensor_height / 2);
//             max_trans_displacement_x[index] = std::max(0.0, velocity_x_i * dt);
//             min_trans_displacement_x[index] = std::min(0.0, velocity_x_i * dt);
//             max_trans_displacement_y[index] = std::max(0.0, velocity_y_i * dt);
//             min_trans_displacement_y[index] = std::min(0.0, velocity_y_i * dt);
//         }
        
//         // Take maximum and minimum of all displacements
//         double max_displacement_x = std::max({
//             *std::max_element(max_rot_displacement_x.begin(), max_rot_displacement_x.end()) +
//             *std::max_element(max_scale_displacement_x.begin(), max_scale_displacement_x.end()) +
//             *std::max_element(max_trans_displacement_x.begin(), max_trans_displacement_x.end()),
//             -(*std::min_element(min_rot_displacement_x.begin(), min_rot_displacement_x.end())) +
//             *std::min_element(min_scale_displacement_x.begin(), min_scale_displacement_x.end()) +
//             *std::min_element(min_trans_displacement_x.begin(), min_trans_displacement_x.end())
//         });
//         double max_displacement_y = std::max({
//             *std::max_element(max_rot_displacement_y.begin(), max_rot_displacement_y.end()) +
//             *std::max_element(max_scale_displacement_y.begin(), max_scale_displacement_y.end()) +
//             *std::max_element(max_trans_displacement_y.begin(), max_trans_displacement_y.end()),
//             -(*std::min_element(min_rot_displacement_y.begin(), min_rot_displacement_y.end())) +
//             *std::min_element(min_scale_displacement_y.begin(), min_scale_displacement_y.end()) +
//             *std::min_element(min_trans_displacement_y.begin(), min_trans_displacement_y.end())
//         });

//         double min_displacement_x = std::min({
//             *std::max_element(max_rot_displacement_x.begin(), max_rot_displacement_x.end()) +
//             *std::max_element(max_scale_displacement_x.begin(), max_scale_displacement_x.end()) +
//             *std::max_element(max_trans_displacement_x.begin(), max_trans_displacement_x.end()),
//             -(*std::min_element(min_rot_displacement_x.begin(), min_rot_displacement_x.end())) +
//             *std::min_element(min_scale_displacement_x.begin(), min_scale_displacement_x.end()) +
//             *std::min_element(min_trans_displacement_x.begin(), min_trans_displacement_x.end())
//         });
//         double min_displacement_y = std::min({
//             *std::max_element(max_rot_displacement_y.begin(), max_rot_displacement_y.end()) +
//             *std::max_element(max_scale_displacement_y.begin(), max_scale_displacement_y.end()) +
//             *std::max_element(max_trans_displacement_y.begin(), max_trans_displacement_y.end()),
//             -(*std::min_element(min_rot_displacement_y.begin(), min_rot_displacement_y.end())) +
//             *std::min_element(min_scale_displacement_y.begin(), min_scale_displacement_y.end()) +
//             *std::min_element(min_trans_displacement_y.begin(), min_trans_displacement_y.end())
//         });

//         const auto width = static_cast<std::size_t>(sensor_width + max_displacement_x - min_displacement_x);
//         const auto height = static_cast<std::size_t>(sensor_height + max_displacement_y - min_displacement_y);

//         const std::array<npy_intp, 2> dimensions{
//             static_cast<npy_intp>(height), static_cast<npy_intp>(width)};
//         auto result = reinterpret_cast<PyArrayObject*>(
//             PyArray_Zeros(2, dimensions.data(), PyArray_DescrFromType(NPY_FLOAT64), 0));
//         Py_BEGIN_ALLOW_THREADS;
//         for (npy_intp index = 0; index < size; ++index) {
//             const auto t = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, index));
//             const auto dt = t - t0;
//             const auto velocity_x_i = *reinterpret_cast<double*>(PyArray_GETPTR1(velocity_x, index));
//             const auto velocity_y_i = *reinterpret_cast<double*>(PyArray_GETPTR1(velocity_y, index));
//             const auto wx_i = *reinterpret_cast<double*>(PyArray_GETPTR1(wx, index));
//             const auto wy_i = *reinterpret_cast<double*>(PyArray_GETPTR1(wy, index));
//             const auto wz_i = *reinterpret_cast<double*>(PyArray_GETPTR1(wz, index));
//             const auto zoom_i = *reinterpret_cast<double*>(PyArray_GETPTR1(zoom, index));
//             const auto x_uncentered = *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index));
//             const auto y_uncentered = *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index));

//             Eigen::Matrix3f rot_mat;
//             rot_mat << 0, -wz_i, wy_i, 
//                     wz_i, 0, -wx_i, 
//                     -wy_i, wx_i, 0;

//             // Center the event points around the middle of the sensor
//             const auto x_centered = x_uncentered - sensor_width / 2.0;
//             const auto y_centered = y_uncentered - sensor_height / 2.0;

//             Eigen::Vector3f event_point(x_centered, y_centered, 1.0);
//             Eigen::Matrix3f transform = Eigen::Matrix3f::Identity();

//             // Apply scaling first
//             Eigen::Matrix3f scale_mat = Eigen::Matrix3f::Identity();
//             scale_mat(0, 0) = 1 - dt * zoom_i;
//             scale_mat(1, 1) = 1 - dt * zoom_i;
//             transform *= scale_mat;

//             // Then apply the rotation
//             Eigen::Matrix3f rot_exp = rot_mat * dt;
//             rot_exp = rot_exp.exp();
//             transform *= rot_exp;

//             // Finally, apply the desired translation
//             transform(0, 2) += dt * velocity_x_i;
//             transform(1, 2) += dt * velocity_y_i;

//             // Translate back to the center of the image
//             transform(0, 2) += sensor_width / 2.0 - min_displacement_x;
//             transform(1, 2) += sensor_height / 2.0 - min_displacement_y;

//             // Apply the transformation matrix to the event point
//             Eigen::Vector3f transformed_point = transform * event_point;
//             const auto x = std::abs(transformed_point[0]);
//             const auto y = std::abs(transformed_point[1]);

//             auto xi = std::floor(x);
//             auto yi = std::floor(y);
//             const auto xf = (x - xi);
//             const auto yf = (y - yi);

//             if ((xi >= 0) && (yi >= 0) && (xi + 1 < width) && (yi + 1 < height)) {
//                 (*reinterpret_cast<double*>(PyArray_GETPTR2(result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi)))) +=
//                     (1.0 - xf) * (1.0 - yf);
//                 (*reinterpret_cast<double*>(PyArray_GETPTR2(result, static_cast<npy_intp>(yi), static_cast<npy_intp>(xi) + 1))) +=
//                     xf * (1.0 - yf);
//                 (*reinterpret_cast<double*>(PyArray_GETPTR2(result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi)))) +=
//                     (1.0 - xf) * yf;
//                 (*reinterpret_cast<double*>(PyArray_GETPTR2(result, static_cast<npy_intp>(yi) + 1, static_cast<npy_intp>(xi) + 1))) +=
//                     xf * yf;
//             }
//         }
//         Py_END_ALLOW_THREADS;
//         return reinterpret_cast<PyObject*>(result);
//     } catch (const std::exception& exception) {
//         PyErr_SetString(PyExc_RuntimeError, exception.what());
//     }
//     return nullptr;
// }

// calculate the variance using the pixel count operation
static PyObject* intensity_variance(PyObject* self, PyObject* args) {
    int32_t sensor_width;
    int32_t sensor_height;
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    double velocity_x;
    double velocity_y;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOdd",
            &sensor_width,
            &sensor_height,
            &raw_ts,
            &raw_xs,
            &raw_ys,
            &velocity_x,
            &velocity_y)) {
        return nullptr;
    }
    try {
        if (sensor_width <= 0) {
            throw std::runtime_error("sensor_width must be larger than zero");
        }
        if (sensor_height <= 0) {
            throw std::runtime_error("sensor_height must be larger than zero");
        }
        if (!PyArray_Check(raw_ts)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
        if (PyArray_NDIM(ts) != 1) {
            throw std::runtime_error("t's dimension must be 1");
        }
        if (PyArray_TYPE(ts) != NPY_FLOAT64) {
            throw std::runtime_error("t's type must be float");
        }
        if (!PyArray_Check(raw_xs)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
        if (PyArray_NDIM(xs) != 1) {
            throw std::runtime_error("x's dimension must be 1");
        }
        if (PyArray_TYPE(xs) != NPY_FLOAT64) {
            throw std::runtime_error("x's type must be float");
        }
        if (!PyArray_Check(raw_ys)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
        if (PyArray_NDIM(ys) != 1) {
            throw std::runtime_error("y's dimension must be 1");
        }
        if (PyArray_TYPE(ys) != NPY_FLOAT64) {
            throw std::runtime_error("y's type must be float");
        }
        const auto size = PyArray_SIZE(ts);
        if (PyArray_SIZE(xs) != size) {
            throw std::runtime_error("t and x must have the same size");
        }
        if (PyArray_SIZE(ys) != size) {
            throw std::runtime_error("t and y must have the same size");
        }
        auto variance = 0.0;
        Py_BEGIN_ALLOW_THREADS;
        const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        const auto maximum_delta_x = std::floor(std::abs(velocity_x) * (t1 - t0)) + 2.0;
        const auto maximum_delta_y = std::floor(std::abs(velocity_y) * (t1 - t0)) + 2.0;
        const auto width  = static_cast<std::size_t>(sensor_width + maximum_delta_x);
        const auto height = static_cast<std::size_t>(sensor_height + maximum_delta_y);
        std::vector<double> cumulative_map(width * height, 0.0);
        for (npy_intp index = 0; index < size; ++index) {
            const auto warped_x =
                *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index))
                - velocity_x * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto warped_y =
                *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index))
                - velocity_y * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto x = warped_x + (velocity_x > 0 ? maximum_delta_x : 0.0);
            const auto y = warped_y + (velocity_y > 0 ? maximum_delta_y : 0.0);
            auto xi = std::floor(x);
            auto yi = std::floor(y);
            const auto xf = x - xi;
            const auto yf = y - yi;
            cumulative_map[xi + yi * width] += (1.0 - xf) * (1.0 - yf);
            cumulative_map[xi + 1 + yi * width] += xf * (1.0 - yf);
            cumulative_map[xi + (yi + 1) * width] += (1.0 - xf) * yf;
            cumulative_map[xi + 1 + (yi + 1) * width] += xf * yf;
        }
        auto mean = 0.0;
        auto m2 = 0.0;
        auto minimum_determinant = 0.0;
        auto maximum_determinant = 0.0;
        auto corrected_velocity_x = 0.0;
        auto corrected_velocity_y = 0.0;
        if ((velocity_x >= 0.0) == (velocity_y >= 0.0)) {
            corrected_velocity_x = std::abs(velocity_x);
            corrected_velocity_y = std::abs(velocity_y);
            minimum_determinant = -corrected_velocity_y * sensor_width;
            maximum_determinant = corrected_velocity_x * sensor_height;
        } else {
            corrected_velocity_x = std::abs(velocity_x);
            corrected_velocity_y = -std::abs(velocity_y);
            minimum_determinant = corrected_velocity_x * maximum_delta_y;
            maximum_determinant = corrected_velocity_x * (maximum_delta_y + sensor_height)
                                  - corrected_velocity_y * sensor_width;
        }
        std::size_t count = 0;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const auto determinant = y * corrected_velocity_x - x * corrected_velocity_y;
                if (determinant >= minimum_determinant && determinant <= maximum_determinant) {
                    const auto value = cumulative_map[x + y * width];
                    const auto delta = value - mean;
                    mean += delta / static_cast<double>(count + 1);
                    m2 += delta * (value - mean);
                    ++count;
                }
            }
        }
        variance = m2 / static_cast<double>(count);
        Py_END_ALLOW_THREADS;
        return PyFloat_FromDouble(variance);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

// calculate the variance of the time surface image
static PyObject* intensity_variance_ts(PyObject* self, PyObject* args) {
    int32_t sensor_width;
    int32_t sensor_height;
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    double velocity_x;
    double velocity_y;
    double tau;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOdd|d",
            &sensor_width,
            &sensor_height,
            &raw_ts,
            &raw_xs,
            &raw_ys,
            &velocity_x,
            &velocity_y,
            &tau)) {
        return nullptr;
    }
    try {
        if (sensor_width <= 0) {
            throw std::runtime_error("sensor_width must be larger than zero");
        }
        if (sensor_height <= 0) {
            throw std::runtime_error("sensor_height must be larger than zero");
        }
        if (!PyArray_Check(raw_ts)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
        if (PyArray_NDIM(ts) != 1) {
            throw std::runtime_error("t's dimension must be 1");
        }
        if (PyArray_TYPE(ts) != NPY_FLOAT64) {
            throw std::runtime_error("t's type must be float");
        }
        if (!PyArray_Check(raw_xs)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
        if (PyArray_NDIM(xs) != 1) {
            throw std::runtime_error("x's dimension must be 1");
        }
        if (PyArray_TYPE(xs) != NPY_FLOAT64) {
            throw std::runtime_error("x's type must be float");
        }
        if (!PyArray_Check(raw_ys)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
        if (PyArray_NDIM(ys) != 1) {
            throw std::runtime_error("y's dimension must be 1");
        }
        if (PyArray_TYPE(ys) != NPY_FLOAT64) {
            throw std::runtime_error("y's type must be float");
        }
        const auto size = PyArray_SIZE(ts);
        if (PyArray_SIZE(xs) != size) {
            throw std::runtime_error("t and x must have the same size");
        }
        if (PyArray_SIZE(ys) != size) {
            throw std::runtime_error("t and y must have the same size");
        }
        auto variance = 0.0;
        Py_BEGIN_ALLOW_THREADS;
        const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        const auto maximum_delta_x = std::floor(std::abs(velocity_x) * (t1 - t0)) + 2.0;
        const auto maximum_delta_y = std::floor(std::abs(velocity_y) * (t1 - t0)) + 2.0;
        const auto width  = static_cast<std::size_t>(sensor_width + maximum_delta_x);
        const auto height = static_cast<std::size_t>(sensor_height + maximum_delta_y);
        std::vector<double> cumulative_map(width * height, 0.0);
        for (npy_intp index = 0; index < size; ++index) {
            const auto warped_x =
                *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index))
                - velocity_x * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto warped_y =
                *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index))
                - velocity_y * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto x = warped_x + (velocity_x > 0 ? maximum_delta_x : 0.0);
            const auto y = warped_y + (velocity_y > 0 ? maximum_delta_y : 0.0);
            auto xi = std::floor(x);
            auto yi = std::floor(y);
            const auto xf = x - xi;
            const auto yf = y - yi;
            if (xi >= 0 && static_cast<std::size_t>(xi) < width && yi >= 0 && static_cast<std::size_t>(yi) < height) {
                double* current_pixel_value = &cumulative_map[xi + yi * width];
                double t_current = *current_pixel_value;

                // Apply decay equation
                double S = std::exp((t_current - (*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index)))) / tau);
                    
                // Update the pixel with the new timestamp
                *current_pixel_value = (*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) * S;
            }
        }
        auto mean = 0.0;
        auto m2 = 0.0;
        auto minimum_determinant = 0.0;
        auto maximum_determinant = 0.0;
        auto corrected_velocity_x = 0.0;
        auto corrected_velocity_y = 0.0;
        if ((velocity_x >= 0.0) == (velocity_y >= 0.0)) {
            corrected_velocity_x = std::abs(velocity_x);
            corrected_velocity_y = std::abs(velocity_y);
            minimum_determinant = -corrected_velocity_y * sensor_width;
            maximum_determinant = corrected_velocity_x * sensor_height;
        } else {
            corrected_velocity_x = std::abs(velocity_x);
            corrected_velocity_y = -std::abs(velocity_y);
            minimum_determinant = corrected_velocity_x * maximum_delta_y;
            maximum_determinant = corrected_velocity_x * (maximum_delta_y + sensor_height)
                                  - corrected_velocity_y * sensor_width;
        }
        std::size_t count = 0;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const auto determinant = y * corrected_velocity_x - x * corrected_velocity_y;
                if (determinant >= minimum_determinant && determinant <= maximum_determinant) {
                    const auto value = cumulative_map[x + y * width];
                    const auto delta = value - mean;
                    mean += delta / static_cast<double>(count + 1);
                    m2 += delta * (value - mean);
                    ++count;
                }
            }
        }
        variance = m2 / static_cast<double>(count);
        Py_END_ALLOW_THREADS;
        return PyFloat_FromDouble(variance);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

static PyObject* intensity_maximum(PyObject* self, PyObject* args) {
    int32_t sensor_width;
    int32_t sensor_height;
    PyObject* raw_ts;
    PyObject* raw_xs;
    PyObject* raw_ys;
    double velocity_x;
    double velocity_y;
    if (!PyArg_ParseTuple(
            args,
            "iiOOOdd",
            &sensor_width,
            &sensor_height,
            &raw_ts,
            &raw_xs,
            &raw_ys,
            &velocity_x,
            &velocity_y)) {
        return nullptr;
    }
    try {
        if (sensor_width <= 0) {
            throw std::runtime_error("sensor_width must be larger than zero");
        }
        if (sensor_height <= 0) {
            throw std::runtime_error("sensor_height must be larger than zero");
        }
        if (!PyArray_Check(raw_ts)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto ts = reinterpret_cast<PyArrayObject*>(raw_ts);
        if (PyArray_NDIM(ts) != 1) {
            throw std::runtime_error("t's dimension must be 1");
        }
        if (PyArray_TYPE(ts) != NPY_FLOAT64) {
            throw std::runtime_error("t's type must be float");
        }
        if (!PyArray_Check(raw_xs)) {
            throw std::runtime_error("t must be a numpy array");
        }
        auto xs = reinterpret_cast<PyArrayObject*>(raw_xs);
        if (PyArray_NDIM(xs) != 1) {
            throw std::runtime_error("x's dimension must be 1");
        }
        if (PyArray_TYPE(xs) != NPY_FLOAT64) {
            throw std::runtime_error("x's type must be float");
        }
        if (!PyArray_Check(raw_ys)) {
            throw std::runtime_error("x must be a numpy array");
        }
        auto ys = reinterpret_cast<PyArrayObject*>(raw_ys);
        if (PyArray_NDIM(ys) != 1) {
            throw std::runtime_error("y's dimension must be 1");
        }
        if (PyArray_TYPE(ys) != NPY_FLOAT64) {
            throw std::runtime_error("y's type must be float");
        }
        const auto size = PyArray_SIZE(ts);
        if (PyArray_SIZE(xs) != size) {
            throw std::runtime_error("t and x must have the same size");
        }
        if (PyArray_SIZE(ys) != size) {
            throw std::runtime_error("t and y must have the same size");
        }
        auto maximum = 0.0;
        Py_BEGIN_ALLOW_THREADS;
        const auto t0 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, 0));
        const auto t1 = *reinterpret_cast<double*>(PyArray_GETPTR1(ts, size - 1));
        const auto maximum_delta_x = std::floor(std::abs(velocity_x) * (t1 - t0)) + 2.0;
        const auto maximum_delta_y = std::floor(std::abs(velocity_y) * (t1 - t0)) + 2.0;
        const auto width = static_cast<std::size_t>(sensor_width + maximum_delta_x);
        const auto height = static_cast<std::size_t>(sensor_height + maximum_delta_y);
        std::vector<double> cumulative_map(width * height, 0.0);
        for (npy_intp index = 0; index < size; ++index) {
            const auto warped_x =
                *reinterpret_cast<double*>(PyArray_GETPTR1(xs, index))
                - velocity_x * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto warped_y =
                *reinterpret_cast<double*>(PyArray_GETPTR1(ys, index))
                - velocity_y * ((*reinterpret_cast<double*>(PyArray_GETPTR1(ts, index))) - t0);
            const auto x = warped_x + (velocity_x > 0 ? maximum_delta_x : 0.0);
            const auto y = warped_y + (velocity_y > 0 ? maximum_delta_y : 0.0);
            auto xi = std::floor(x);
            auto yi = std::floor(y);
            const auto xf = x - xi;
            const auto yf = y - yi;
            cumulative_map[xi + yi * width] += (1.0 - xf) * (1.0 - yf);
            maximum = std::max(maximum, cumulative_map[xi + yi * width]);
            cumulative_map[xi + 1 + yi * width] += xf * (1.0 - yf);
            maximum = std::max(maximum, cumulative_map[xi + 1 + yi * width]);
            cumulative_map[xi + (yi + 1) * width] += (1.0 - xf) * yf;
            maximum = std::max(maximum, cumulative_map[xi + (yi + 1) * width]);
            cumulative_map[xi + 1 + (yi + 1) * width] += xf * yf;
            maximum = std::max(maximum, cumulative_map[xi + 1 + (yi + 1) * width]);
        }
        Py_END_ALLOW_THREADS;
        return PyFloat_FromDouble(maximum);
    } catch (const std::exception& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
    }
    return nullptr;
}

static PyMethodDef event_warping_extension_methods[] = {
    {"smooth_histogram", smooth_histogram, METH_VARARGS, nullptr},
    {"accumulate", accumulate, METH_VARARGS, nullptr},
    {"accumulate_timesurface", accumulate_timesurface, METH_VARARGS, nullptr},
    {"accumulate_pixel_map", accumulate_pixel_map, METH_VARARGS, nullptr},
    {"accumulate_cnt", accumulate_cnt, METH_VARARGS, nullptr},
    {"accumulate_cnt_rgb", accumulate_cnt_rgb, METH_VARARGS, nullptr},
    // {"geometricTransformation", geometricTransformation, METH_VARARGS, nullptr},
    // {"accumulate4D", accumulate4D, METH_VARARGS, nullptr},
    // {"accumulate4D_cnt", accumulate4D_cnt, METH_VARARGS, nullptr},
    {"intensity_variance", intensity_variance, METH_VARARGS, nullptr},
    {"intensity_variance_ts", intensity_variance_ts, METH_VARARGS, nullptr},
    {"intensity_maximum", intensity_maximum, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};
static struct PyModuleDef event_warping_extension_definition = {
    PyModuleDef_HEAD_INIT,
    "event_warping_extension",
    "event_warping_extension speeds up some sessiontools operations",
    -1,
    event_warping_extension_methods};
PyMODINIT_FUNC PyInit_event_warping_extension() {
    auto module = PyModule_Create(&event_warping_extension_definition);
    import_array();
    return module;
}
