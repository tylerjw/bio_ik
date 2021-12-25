// Copyright (c) 2016-2017, Philipp Sebastian Ruppel
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universität Hamburg nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <fann.h>
#include <fann_cpp.h>

#include <mutex>

#include "ik_base.h"

#define LOG_LIST(l)                                \
  {                                                \
    LOG(#l "[]");                                  \
    for (std::size_t i = 0; i < l.size(); i++) {   \
      LOG(#l "[" + std::to_string(i) + "]", l[i]); \
    }                                              \
  }

namespace bio_ik {

/*
static inline KDL::Twist toTwist(const Frame& frame)
{
    KDL::Twist t;
    t.vel.x(frame.pos.x());
    t.vel.y(frame.pos.y());
    t.vel.z(frame.pos.z());
    auto r = frame.rot.getAxis() * frame.rot.getAngle();
    t.rot.x(r.x());
    t.rot.y(r.y());
    t.rot.z(r.z());
    return t;
}

static inline KDL::Twist frameTwist(const Frame& a, const Frame& b)
{
    auto frame = inverse(a) * b;
    KDL::Twist t;
    t.vel.x(frame.pos.x());
    t.vel.y(frame.pos.y());
    t.vel.z(frame.pos.z());
    auto r = frame.rot.getAxis() * frame.rot.getAngle();
    t.rot.x(r.x());
    t.rot.y(r.y());
    t.rot.z(r.z());
    return t;
}
*/

struct IKNeural : IKBase {
  std::vector<double> solution_;
  FANN::neural_net net_;

  std::vector<std::pair<fann_type, fann_type>> input_minmax_, output_minmax_;

  static void find_minmax(
      const std::vector<fann_type>& values,
      std::vector<std::pair<fann_type, fann_type>>& minmax) {
    for (size_t i = 0; i < minmax.size(); i++) {
      minmax[i] = std::make_pair(values[i], values[i]);
    }
    for (size_t i = 0; i < values.size(); i++) {
      auto& v = values[i];
      auto& p = minmax[i % minmax.size()];
      p.first = std::min(p.first, v);
      p.second = std::max(p.second, v);
    }
  }

  static void normalize(
      std::vector<fann_type>& values,
      const std::vector<std::pair<fann_type, fann_type>>& minmax) {
    for (size_t i = 0; i < values.size(); i++) {
      auto& v = values[i];
      auto& p = minmax[i % minmax.size()];
      v = (v - p.first) / (p.second - p.first);
    }
  }

  static void denormalize(
      std::vector<fann_type>& values,
      const std::vector<std::pair<fann_type, fann_type>>& minmax) {
    for (size_t i = 0; i < values.size(); i++) {
      auto& v = values[i];
      auto& p = minmax[i % minmax.size()];
      v = v * (p.second - p.first) + p.first;
    }
  }

  unsigned int input_count_, output_count_;

  IKNeural(const IKParams& p) : IKBase(p) { trained_ = false; }

  bool trained_;

  void train() {
    trained_ = true;

    input_count_ =
        problem_.active_variables.size() + problem_.tip_link_indices.size() * 6;
    output_count_ = problem_.active_variables.size();

    LOG_VAR(input_count_);
    LOG_VAR(output_count_);

    // std::vector<unsigned int> levels = {input_count_, input_count_,
    // input_count_, output_count_};

    // std::vector<unsigned int> levels = {input_count_, input_count_ +
    // output_count_, output_count_};

    // std::vector<unsigned int> levels = {input_count_, input_count_ +
    // output_count_, input_count_ + output_count_, output_count_};

    // std::vector<unsigned int> levels = {input_count_, 100, output_count_};

    std::vector<unsigned int> levels = {input_count_, 50, output_count_};

    net_.create_standard_array(levels.size(), levels.data());

    size_t var_count = params_.robot_model->getVariableCount();
    std::vector<double> state1(var_count), state2(var_count);
    std::vector<Frame> frames1, frames2;

    std::vector<fann_type> inputs, outputs;
    std::vector<fann_type*> input_pp, output_pp;

    LOG("neuro ik generating training data");

    unsigned int samples = 10000;

    for (size_t iter = 0; iter < samples; iter++) {
      for (size_t ivar = 0; ivar < var_count; ivar++) {
        state1[ivar] = random(modelInfo_.getMin(ivar), modelInfo_.getMax(ivar));
        state1[ivar] = modelInfo_.clip(state1[ivar], ivar);
        // state2[ivar] = modelInfo_.clip(state1[ivar] + random_gauss() *
        // modelInfo_.getSpan(ivar), ivar);
        state2[ivar] =
            modelInfo_.clip(state1[ivar] + random_gauss() * 0.1, ivar);
      }

      model_.applyConfiguration(state1);
      frames1 = model_.getTipFrames();
      model_.applyConfiguration(state2);
      frames2 = model_.getTipFrames();

      for (auto ivar : problem_.active_variables) {
        inputs.push_back(state1[ivar]);
        outputs.push_back(state2[ivar] - state1[ivar]);
      }

      for (size_t itip = 0; itip < problem_.tip_link_indices.size(); itip++) {
        double translational_scale = 1.0;
        double rotational_scale = 1.0;

        // Frame diff = inverse(frames1[itip]) * frames2[itip];
        // auto twist = toTwist(diff);
        auto twist = frameTwist(frames1[itip], frames2[itip]);

        inputs.push_back(frames2[itip].pos.x() - frames1[itip].pos.x());
        inputs.push_back(frames2[itip].pos.y() - frames1[itip].pos.y());
        inputs.push_back(frames2[itip].pos.z() - frames1[itip].pos.z());

        inputs.push_back(twist.rot.x() * rotational_scale);
        inputs.push_back(twist.rot.y() * rotational_scale);
        inputs.push_back(twist.rot.z() * rotational_scale);
      }
    }

    for (auto& v : inputs)
      if (!std::isfinite(v)) throw std::runtime_error("NAN");
    for (auto& v : outputs)
      if (!std::isfinite(v)) throw std::runtime_error("NAN");

    input_minmax_.resize(input_count_);
    output_minmax_.resize(output_count_);

    find_minmax(inputs, input_minmax_);
    find_minmax(outputs, output_minmax_);

    normalize(inputs, input_minmax_);
    normalize(outputs, output_minmax_);

    for (size_t iter = 0; iter < samples; iter++) {
      input_pp.push_back(inputs.data() + iter * input_count_);
      output_pp.push_back(outputs.data() + iter * output_count_);
    }

    LOG("neuro ik training");

    FANN::training_data train;
    train.set_train_data(samples, input_count_, input_pp.data(), output_count_,
                         output_pp.data());
    net_.set_callback(
        [](FANN::neural_net& net, FANN::training_data& train,
           unsigned int max_epochs, unsigned int epochs_between_reports,
           float desired_error, unsigned int epochs, void* user_data) {
          if (epochs % epochs_between_reports != 0) return 0;
          // LOG("training", epochs, "/", max_epochs, epochs * 100 / max_epochs,
          // "%");
          LOG("training", epochs, net.get_MSE(), desired_error);
          return 0;
        },
        0);

    net_.set_activation_function_hidden(FANN::SIGMOID);
    net_.set_activation_function_output(FANN::SIGMOID);

    net_.init_weights(train);

    net_.train_on_data(train, 1000, 1, 0.0001);

    fann_type err = net_.test_data(train);
    LOG("neuro ik training error:", err);

    /*std::vector<fann_type> iiv, oov, ttv;
    for(size_t iter = 0; iter < 10; iter++)
    {
        auto* ii = input_pp[iter];
        auto* oo = net_.run(ii);
        auto* tt = output_pp[iter];
        iiv.assign(ii, ii + input_count_);
        ttv.assign(tt, tt + output_count_);
        oov.assign(oo, oo + output_count_);
        LOG_LIST(iiv);
        LOG_LIST(ttv);
        LOG_LIST(oov);
    }*/

    LOG("training done");
  }

  size_t iterations_ = 0;

  void initialize(const Problem& problem) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    IKBase::initialize(problem);
    solution_ = problem_.initial_guess;
    if (!trained_) train();
    iterations_ = 0;
    if (thread_index_ > 0)
      for (auto& vi : problem_.active_variables)
        solution_[vi] = random(modelInfo_.getMin(vi), modelInfo_.getMax(vi));
  }

  const std::vector<double>& getSolution() const { return solution_; }

  std::vector<fann_type> inputs_, outputs_;

  std::vector<Frame> tip_objectives_;

  /*void step()
  {
      //if(iterations_ > 1) return;
      iterations_++;

      inputs_.clear();
      for(auto ivar : problem_.active_variables)
      {
          inputs_.push_back(solution_[ivar]);
      }

      tip_objectives_.resize(model_.getTipFrames().size());
      for(auto& g : problem_.goals)
      {
          tip_objectives_[g.tip_index] = g.frame;
      }

      model_.applyConfiguration(solution_);
      auto& frames1 = model_.getTipFrames();
      auto& frames2 = tip_objectives_;

      double scale = 1.0;

      for(size_t itip = 0; itip < tip_objectives_.size(); itip++)
      {
          double translational_scale = 1.0;
          double rotational_scale = 1.0;

          //Frame diff = inverse(frames1[itip]) * frames2[itip];
          //auto twist = toTwist(diff);
          auto twist = frameTwist(frames1[itip], frames2[itip]);

          auto dpos = frames2[itip].pos - frames1[itip].pos;
          auto drot = Vector3(
              twist.rot.x() * rotational_scale,
              twist.rot.y() * rotational_scale,
              twist.rot.z() * rotational_scale
          );

          scale = 1.0 / (0.0000001 + dpos.length() + drot.length());

          dpos = dpos * scale;
          drot = drot * scale;

          inputs_.push_back(dpos.x());
          inputs_.push_back(dpos.y());
          inputs_.push_back(dpos.z());

          inputs_.push_back(drot.x());
          inputs_.push_back(drot.y());
          inputs_.push_back(drot.z());
      }

      normalize(inputs_, input_minmax_);

      auto* oo = net_.run(inputs_.data());

      outputs_.assign(oo, oo + output_count_);

      denormalize(outputs_, output_minmax_);

      auto& vv = problem_.active_variables;
      for(size_t iout = 0; iout < vv.size(); iout++)
      {
          size_t ivar = vv[iout];
          solution_[ivar] = modelInfo_.clip(solution_[ivar] + outputs_[iout] *
  0.1 / scale, ivar);
      }
  }*/

  void step() {
    // if(iterations_ > 1) return;
    iterations_++;

    inputs_.clear();
    for (auto ivar : problem_.active_variables) {
      inputs_.push_back(solution_[ivar]);
    }

    tip_objectives_.resize(model_.getTipFrames().size());
    for (auto& g : problem_.goals) {
      tip_objectives_[g.tip_index] = g.frame;
    }

    model_.applyConfiguration(solution_);
    auto& frames1 = model_.getTipFrames();
    auto& frames2 = tip_objectives_;

    double scale = 1.0;
    for (size_t itip = 0; itip < tip_objectives_.size(); itip++) {
      double translational_scale = 1.0;
      double rotational_scale = 1.0;
      auto twist = frameTwist(frames1[itip], frames2[itip]);
      auto dpos = frames2[itip].pos - frames1[itip].pos;
      auto drot = Vector3(twist.rot.x() * rotational_scale,
                          twist.rot.y() * rotational_scale,
                          twist.rot.z() * rotational_scale);

      /*if(iterations_ % 2)
      {
          scale = 1.0 / (0.0000001 + dpos.length());
          inputs_.push_back(dpos.x() * scale);
          inputs_.push_back(dpos.y() * scale);
          inputs_.push_back(dpos.z() * scale);
          inputs_.push_back(0);
          inputs_.push_back(0);
          inputs_.push_back(0);
      } else {
          scale = 1.0 / (0.0000001 + drot.length());
          inputs_.push_back(0);
          inputs_.push_back(0);
          inputs_.push_back(0);
          inputs_.push_back(drot.x() * scale);
          inputs_.push_back(drot.y() * scale);
          inputs_.push_back(drot.z() * scale);
      }*/

      {
        scale = 1.0 / (0.0000001 + dpos.length() + drot.length());
        inputs_.push_back(dpos.x() * scale);
        inputs_.push_back(dpos.y() * scale);
        inputs_.push_back(dpos.z() * scale);
        inputs_.push_back(drot.x() * scale);
        inputs_.push_back(drot.y() * scale);
        inputs_.push_back(drot.z() * scale);
      }
    }
    normalize(inputs_, input_minmax_);
    auto* oo = net_.run(inputs_.data());
    outputs_.assign(oo, oo + output_count_);
    denormalize(outputs_, output_minmax_);
    auto& vv = problem_.active_variables;
    for (size_t iout = 0; iout < vv.size(); iout++) {
      size_t ivar = vv[iout];
      solution_[ivar] =
          modelInfo_.clip(solution_[ivar] + outputs_[iout] * 1 / scale, ivar);
    }
  }
};

static IKFactory::Class<IKNeural> neural("neural");

struct IKNeural2 : IKBase {
  std::vector<double> solution_;
  FANN::neural_net net_;

  std::vector<std::pair<fann_type, fann_type>> input_minmax_, output_minmax_;

  /*static void find_minmax(const std::vector<fann_type>& values,
  std::vector<std::pair<fann_type, fann_type>>& minmax)
  {
      for(size_t i = 0; i < minmax.size(); i++)
      {
          minmax[i] = std::make_pair(values[i], values[i]);
      }
      for(size_t i = 0; i < values.size(); i++)
      {
          auto& v = values[i];
          auto& p = minmax[i % minmax.size()];
          p.first = std::min(p.first, v);
          p.second = std::max(p.second, v);
      }
  }*/

  static void find_minmax(
      const std::vector<fann_type>& values,
      std::vector<std::pair<fann_type, fann_type>>& minmax) {
    std::vector<double> centers(minmax.size(), 0.0);
    for (size_t i = 0; i < values.size(); i++)
      centers[i % minmax.size()] +=
          values[i] * (1.0 * minmax.size() / values.size());

    std::vector<double> ranges2(minmax.size(), 0.0001);
    for (size_t i = 0; i < values.size(); i++) {
      double d = values[i] - centers[i % minmax.size()];
      d = d * d;
      ranges2[i % minmax.size()] += d * (1.0 * minmax.size() / values.size());
    }

    for (size_t i = 0; i < minmax.size(); i++) {
      auto& p = minmax[i];
      p.first = centers[i] - sqrt(ranges2[i]);
      p.second = centers[i] + sqrt(ranges2[i]);
    }
  }

  static void normalize(
      std::vector<fann_type>& values,
      const std::vector<std::pair<fann_type, fann_type>>& minmax) {
    for (size_t i = 0; i < values.size(); i++) {
      auto& v = values[i];
      auto& p = minmax[i % minmax.size()];
      v = (v - p.first) / (p.second - p.first);
    }
  }

  static void denormalize(
      std::vector<fann_type>& values,
      const std::vector<std::pair<fann_type, fann_type>>& minmax) {
    for (size_t i = 0; i < values.size(); i++) {
      auto& v = values[i];
      auto& p = minmax[i % minmax.size()];
      v = v * (p.second - p.first) + p.first;
    }
  }

  unsigned int input_count_, output_count_;

  IKNeural2(const IKParams& p) : IKBase(p) { trained_ = false; }

  bool trained_;

  void train() {
    trained_ = true;

    input_count_ = problem_.tip_link_indices.size() * 7;
    output_count_ = problem_.active_variables.size();

    LOG_VAR(input_count_);
    LOG_VAR(output_count_);

    // std::vector<unsigned int> levels = {input_count_, 100, 100,
    // output_count_};

    // std::vector<unsigned int> levels = {input_count_, input_count_,
    // input_count_, output_count_};

    std::vector<unsigned int> levels = {
        input_count_, input_count_ + output_count_, output_count_};

    // std::vector<unsigned int> levels = {input_count_, input_count_ +
    // output_count_, input_count_ + output_count_, output_count_};

    // std::vector<unsigned int> levels = {input_count_, output_count_};

    net_.create_standard_array(levels.size(), levels.data());

    size_t var_count = params_.robot_model->getVariableCount();
    std::vector<double> state = problem_.initial_guess;
    std::vector<Frame> frames;

    std::vector<fann_type> inputs, outputs;
    std::vector<fann_type*> input_pp, output_pp;

    LOG("neuro ik generating training data");

    unsigned int samples = 10000;

    for (size_t iter = 0; iter < samples; iter++) {
      for (size_t ivar : problem_.active_variables)
        state[ivar] = random(modelInfo_.getMin(ivar), modelInfo_.getMax(ivar));

      model_.applyConfiguration(state);
      frames = model_.getTipFrames();

      for (auto ivar : problem_.active_variables)
        outputs.push_back(state[ivar]);

      for (size_t itip = 0; itip < problem_.tip_link_indices.size(); itip++) {
        inputs.push_back(frames[itip].pos.x());
        inputs.push_back(frames[itip].pos.y());
        inputs.push_back(frames[itip].pos.z());

        auto rot = frames[itip].rot;
        rot = rot * rot;
        // rot = tf2::Quaternion(0, 0, 0, 1);
        inputs.push_back(rot.x());
        inputs.push_back(rot.y());
        inputs.push_back(rot.z());
        inputs.push_back(rot.w());
      }
    }

    for (auto& v : inputs)
      if (!std::isfinite(v)) throw std::runtime_error("NAN");
    for (auto& v : outputs)
      if (!std::isfinite(v)) throw std::runtime_error("NAN");

    input_minmax_.resize(input_count_);
    output_minmax_.resize(output_count_);

    find_minmax(inputs, input_minmax_);
    find_minmax(outputs, output_minmax_);

    normalize(inputs, input_minmax_);
    normalize(outputs, output_minmax_);

    for (size_t iter = 0; iter < samples; iter++) {
      input_pp.push_back(inputs.data() + iter * input_count_);
      output_pp.push_back(outputs.data() + iter * output_count_);
    }

    LOG("neuro ik training");

    FANN::training_data train;
    train.set_train_data(samples, input_count_, input_pp.data(), output_count_,
                         output_pp.data());
    net_.set_callback(
        [](FANN::neural_net& net, FANN::training_data& train,
           unsigned int max_epochs, unsigned int epochs_between_reports,
           float desired_error, unsigned int epochs, void* user_data) {
          if (epochs % epochs_between_reports != 0) return 0;
          // LOG("training", epochs, "/", max_epochs, epochs * 100 / max_epochs,
          // "%");
          LOG("training", epochs, net.get_MSE(), desired_error);
          return 0;
        },
        0);

    net_.set_activation_function_hidden(FANN::SIGMOID);
    net_.set_activation_function_output(FANN::SIGMOID);

    net_.init_weights(train);

    net_.train_on_data(train, 100, 1, 0.0001);

    fann_type err = net_.test_data(train);
    LOG("neuro ik training error:", err);

    /*std::vector<fann_type> iiv, oov, ttv;
    for(size_t iter = 0; iter < 10; iter++)
    {
        auto* ii = input_pp[iter];
        auto* oo = net_.run(ii);
        auto* tt = output_pp[iter];
        iiv.assign(ii, ii + input_count_);
        ttv.assign(tt, tt + output_count_);
        oov.assign(oo, oo + output_count_);
        LOG_LIST(iiv);
        LOG_LIST(ttv);
        LOG_LIST(oov);
    }*/

    LOG("training done");
  }

  size_t iterations_ = 0;

  void initialize(const Problem& problem) {
    IKBase::initialize(problem);
    solution_ = problem_.initial_guess;
    if (!trained_) train();
    iterations_ = 0;
  }

  const std::vector<double>& getSolution() const { return solution_; }

  std::vector<fann_type> inputs_, outputs_;

  std::vector<Frame> tip_objectives_;

  void step() {
    if (iterations_ > 1) return;
    iterations_++;

    inputs_.clear();

    tip_objectives_.resize(model_.getTipFrames().size());
    for (auto& g : problem_.goals) {
      tip_objectives_[g.tip_index] = g.frame;
    }

    auto& frames = tip_objectives_;

    for (size_t itip = 0; itip < tip_objectives_.size(); itip++) {
      inputs_.push_back(frames[itip].pos.x());
      inputs_.push_back(frames[itip].pos.y());
      inputs_.push_back(frames[itip].pos.z());

      auto rot = frames[itip].rot;
      rot = rot * rot;
      // rot = tf2::Quaternion(0, 0, 0, 1);
      inputs_.push_back(rot.x());
      inputs_.push_back(rot.y());
      inputs_.push_back(rot.z());
      inputs_.push_back(rot.w());
    }

    normalize(inputs_, input_minmax_);

    auto* oo = net_.run(inputs_.data());

    outputs_.assign(oo, oo + output_count_);

    denormalize(outputs_, output_minmax_);

    auto& vv = problem_.active_variables;
    for (size_t iout = 0; iout < vv.size(); iout++) {
      size_t ivar = vv[iout];
      solution_[ivar] = modelInfo_.clip(outputs_[iout], ivar);
    }
  }
};

static IKFactory::Class<IKNeural2> neural2("neural2");
}  // namespace bio_ik
