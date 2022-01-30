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

#include "bio_ik/ik_evolution_2.hpp"

#include <math.h>                            // for fabs
#include <moveit/robot_model/joint_model.h>  // for JointModel, JointModel::...
#include <moveit/robot_model/robot_model.h>  // for RobotModel, RobotModelCo...
#include <stddef.h>                          // for size_t

#include <algorithm>                      // for sort
#include <bio_ik/forward_kinematics.hpp>  // for RobotFK
#include <bio_ik/ik_base.hpp>             // for IKSolver
#include <bio_ik/problem.hpp>             // for Problem
#include <bio_ik/utils.hpp>               // for aligned_vector, BLOCKPRO...
#include <memory>                         // for allocator, __shared_ptr_...
#include <set>
#include <string>
#include <utility>  // for swap
#include <vector>   // for vector, vector::size_type

#include "bio_ik/frame.hpp"       // for Frame, normalizeFast
#include "bio_ik/robot_info.hpp"  // for RobotInfo

namespace bio_ik {

// fast evolutionary inverse kinematics
template <int memetic>
struct IKEvolution2 : IKSolver {
  struct Individual {
    aligned_vector<double> genes;
    aligned_vector<double> gradients;
    double fitness;
  };

  struct Species {
    std::vector<Individual> individuals;
    double fitness;
    bool improved;
  };

  std::vector<double> initial_guess_;
  std::vector<double> solution_;
  std::vector<double> temp_joint_variables_;
  double solution_fitness_;
  std::vector<Species> species_;
  std::vector<Individual> children_;
  std::vector<aligned_vector<Frame>> phenotypes_, phenotypes2_, phenotypes3_;
  std::vector<size_t> child_indices_;
  std::vector<double*> genotypes_;
  std::vector<Frame> phenotype_;
  std::vector<size_t> quaternion_genes_;
  aligned_vector<double> genes_min_, genes_max_, genes_span_;
  aligned_vector<double> gradient_, temp_;

  IKEvolution2(const IKParams& p) : IKSolver(p) {}

  void genesToJointVariables(const Individual& individual,
                             std::vector<double>& variables) {
    auto& genes = individual.genes;
    variables.resize(params_.robot_model->getVariableCount());
    for (size_t i = 0; i < problem_.active_variables.size(); i++)
      variables[problem_.active_variables[i]] = genes[i];
  }

  const std::vector<double>& getSolution() const { return solution_; }

  void initialize(const Problem& local_problem) {
    BLOCKPROFILER("initialization");

    IKSolver::initialize(local_problem);

    // init list of quaternion joint genes to be normalized during each mutation
    quaternion_genes_.clear();
    for (size_t igene = 0; igene < local_problem.active_variables.size();
         igene++) {
      size_t ivar = local_problem.active_variables[igene];
      auto* joint_model =
          params_.robot_model->getJointOfVariable(static_cast<int>(ivar));
      if (joint_model->getFirstVariableIndex() + 3 != ivar) continue;
      if (joint_model->getType() != moveit::core::JointModel::FLOATING)
        continue;
      quaternion_genes_.push_back(igene);
    }

    // set solution to initial guess
    initial_guess_ = local_problem.initial_guess;
    solution_ = initial_guess_;
    solution_fitness_ = computeFitness(solution_);

    // init temporary buffer with positions of inactive joints
    temp_joint_variables_ = initial_guess_;

    // params
    size_t population_size = 2;
    size_t child_count = 16;

    // initialize population on current island
    species_.resize(2);
    for (auto& s : species_) {
      // create individuals
      s.individuals.resize(population_size);

      // initialize first individual
      {
        auto& v = s.individuals[0];

        // init genes
        v.genes.resize(local_problem.active_variables.size());
        // if(thread_index_ == 0) // on first island?
        // if(thread_index_ % 2 == 0) // on every second island...
        if (1) {
          // set to initial_guess_
          for (size_t i = 0; i < v.genes.size(); i++)
            v.genes[i] = initial_guess_[local_problem.active_variables[i]];
        } else {
          // initialize populations on other islands randomly
          for (size_t i = 0; i < v.genes.size(); i++)
            v.genes[i] =
                random(modelInfo_.getMin(local_problem.active_variables[i]),
                       modelInfo_.getMax(local_problem.active_variables[i]));
        }

        // set gradients to zero
        v.gradients.clear();
        v.gradients.resize(local_problem.active_variables.size(), 0);
      }

      // clone first individual
      for (size_t i = 1; i < s.individuals.size(); i++) {
        s.individuals[i].genes = s.individuals[0].genes;
        s.individuals[i].gradients = s.individuals[0].gradients;
      }
    }

    // space for child population
    children_.resize(population_size + child_count);
    for (auto& child : children_) {
      child.genes.resize(local_problem.active_variables.size());
      child.gradients.resize(local_problem.active_variables.size());
    }

    // init gene infos
    // if(genes_min_.empty())
    {
      genes_min_.resize(local_problem.active_variables.size());
      genes_max_.resize(local_problem.active_variables.size());
      genes_span_.resize(local_problem.active_variables.size());
      for (size_t i = 0; i < local_problem.active_variables.size(); i++) {
        genes_min_[i] =
            modelInfo_.getClipMin(local_problem.active_variables[i]);
        genes_max_[i] =
            modelInfo_.getClipMax(local_problem.active_variables[i]);
        genes_span_[i] = modelInfo_.getSpan(local_problem.active_variables[i]);
      }
    }

    /*
    // init chain mutation masks
    chain_mutation_masks.resize(chain_mutation_mask_count);
    for(auto& chain_mutation_mask : chain_mutation_masks)
    {
        temp_mutation_chain.clear();
        if(local_problem.tip_link_indices.size() > 1)
        {
            for(auto* chain_link =
    params_.robot_model->getLinkModel(random_element(local_problem.tip_link_indices));
    chain_link; chain_link = chain_link->getParentLinkModel())
                temp_mutation_chain.push_back(chain_link);
            temp_mutation_chain.resize(random_index(temp_mutation_chain.size())
    + 1);
        }

        temp_chain_mutation_var_mask.resize(params_.robot_model->getVariableCount());
        for(auto& m : temp_chain_mutation_var_mask) m = 0;
        for(auto* chain_link : temp_mutation_chain)
        {
            auto* chain_joint = chain_link->getParentJointModel();
            for(size_t ivar = chain_joint->getFirstVariableIndex(); ivar <
    chain_joint->getFirstVariableIndex() + chain_joint->getVariableCount();
    ivar++) temp_chain_mutation_var_mask[ivar] = 1;
        }

        chain_mutation_mask.resize(local_problem.active_variables.size());
        for(size_t igene = 0; igene < local_problem.active_variables.size();
    igene++) chain_mutation_mask[igene] =
    temp_chain_mutation_var_mask[local_problem.active_variables[igene]];
    }
    */
  }

  /*
  const size_t chain_mutation_mask_count = 256;
  std::vector<std::vector<int>> chain_mutation_masks;
  std::vector<const moveit::core::LinkModel*> temp_mutation_chain;
  std::vector<int> temp_chain_mutation_var_mask;
  */

  // aligned_vector<double> rmask;

  // create offspring and mutate
  __attribute__((hot)) __attribute__((noinline))
  //__attribute__((target_clones("avx2", "avx", "sse2", "default")))
  //__attribute__((target("avx")))
  void
  reproduce(const std::vector<Individual>& population) {
    const auto __attribute__((aligned(32)))* __restrict__ local_genes_span =
        this->genes_span_.data();
    const auto __attribute__((aligned(32)))* __restrict__ local_genes_min =
        this->genes_min_.data();
    const auto __attribute__((aligned(32)))* __restrict__ local_genes_max =
        this->genes_max_.data();

    auto gene_count = children_[0].genes.size();

    size_t s = (children_.size() - population.size()) * gene_count +
               children_.size() * 4 + 4;

    auto* __restrict__ rr = fast_random_gauss_n(s);
    rr = reinterpret_cast<const double*>((reinterpret_cast<size_t>(rr) + 3) /
                                         4 * 4);

    /*rmask.resize(s);
    for(auto& m : rmask) m = fast_random() < 0.1 ? 1.0 : 0.0;
    double* dm = rmask.data();*/

    for (size_t child_index = population.size(); child_index < children_.size();
         child_index++) {
      double mutation_rate = (1 << fast_random_index(16)) * (1.0 / (1 << 23));
      auto& parent = population[0];
      auto& parent2 = population[1];
      double fmix = (child_index % 2 == 0) * 0.2;
      double gradient_factor = static_cast<double>(child_index % 3);

      auto __attribute__((aligned(32)))* __restrict__ parent_genes =
          parent.genes.data();
      auto __attribute__((aligned(32)))* __restrict__ parent_gradients =
          parent.gradients.data();

      auto __attribute__((aligned(32))) * __restrict__ __attribute__((unused))
                                          parent2_genes = parent2.genes.data();
      auto __attribute__((aligned(32)))* __restrict__ parent2_gradients =
          parent2.gradients.data();

      auto& child = children_[child_index];

      auto __attribute__((aligned(32)))* __restrict__ child_genes =
          child.genes.data();
      auto __attribute__((aligned(32)))* __restrict__ child_gradients =
          child.gradients.data();

      /*
      // TODO(tylerjw): learn performance pragmas
      // #pragma omp simd aligned(local_genes_span : 32),
      aligned(local_genes_min : 32), \
      //     aligned(local_genes_max : 32), aligned(parent_genes : 32), \
      //     aligned(parent_gradients : 32), aligned(parent2_genes : 32),    \
      //     aligned(parent2_gradients : 32), aligned(child_genes : 32),     \
      //     aligned(child_gradients : 32) aligned(rr : 32)
      */

      // TODO(tylerjw): learn performance pragmas
      // #pragma unroll
      for (size_t gene_index = 0; gene_index < gene_count; gene_index++) {
        // double mutation_rate = (1 << fast_random_index(16)) * (1.0 / (1 <<
        // 23));

        double r = rr[gene_index];
        // r *= dm[gene_index];
        double f = mutation_rate * local_genes_span[gene_index];
        double gene = parent_genes[gene_index];
        double parent_gene = gene;
        gene += r * f;
        double parent_gradient = mix(parent_gradients[gene_index],
                                     parent2_gradients[gene_index], fmix);
        double local_gradient = parent_gradient * gradient_factor;
        gene += local_gradient;
        gene = clamp(gene, local_genes_min[gene_index],
                     local_genes_max[gene_index]);
        child_genes[gene_index] = gene;
        child_gradients[gene_index] =
            mix(parent_gradient, gene - parent_gene, 0.3);
      }
      rr += (gene_count + 3) / 4 * 4;
      // dm += (gene_count + 3) / 4 * 4;

      /*if(problem_.tip_link_indices.size() > 1)
      {
          if(fast_random() < 0.5)
          {
              auto& mask =
      chain_mutation_masks[fast_random_index(chain_mutation_mask_count)];
              for(size_t gene_index = 0; gene_index < gene_count; gene_index++)
              {
                  if(!mask[gene_index])
                  {
                      child_genes[gene_index] = parent_genes[gene_index];
                      child_gradients[gene_index] =
      parent_gradients[gene_index];
                  }
              }
          }
      }*/

      for (auto quaternion_gene_index : quaternion_genes_) {
        auto& qpos = (*reinterpret_cast<Quaternion*>(
            &children_[child_index].genes[quaternion_gene_index]));
        normalizeFast(qpos);
      }
    }
  }

  void step() {
    FNPROFILER();

    for (size_t ispecies = 0; ispecies < species_.size(); ispecies++) {
      auto& single_species = species_[ispecies];
      auto& population = single_species.individuals;

      {
        BLOCKPROFILER("evolution");

        // initialize forward kinematics approximator
        genesToJointVariables(single_species.individuals[0],
                              temp_joint_variables_);
        {
          BLOCKPROFILER("fk");
          model_.applyConfiguration(temp_joint_variables_);
          model_.initializeMutationApproximator(problem_.active_variables);
        }

        // run evolution for a few generations
        size_t generation_count = 16;
        if (memetic) generation_count = 8;
        for (size_t generation = 0; generation < generation_count;
             generation++) {
          // BLOCKPROFILER("evolution");

          if (canceled_) break;

          // reproduction
          {
            BLOCKPROFILER("reproduction");
            reproduce(population);
          }

          size_t child_count = children_.size();

          // pre-selection by secondary objectives
          if (problem_.secondary_goals.size()) {
            BLOCKPROFILER("pre-selection");
            child_count =
                random_index(children_.size() - population.size() - 1) + 1 +
                population.size();
            for (size_t child_index = population.size();
                 child_index < children_.size(); child_index++) {
              children_[child_index].fitness =
                  computeSecondaryFitnessActiveVariables(
                      children_[child_index].genes.data());
            }
            {
              BLOCKPROFILER("pre-selection sort");
              std::sort(
                  children_.begin() + static_cast<long int>(population.size()),
                  children_.end(),
                  [](const Individual& a, const Individual& b) {
                    return a.fitness < b.fitness;
                  });
            }
          }

          // keep parents
          {
            BLOCKPROFILER("keep alive");
            for (size_t i = 0; i < population.size(); i++) {
              children_[i].genes = population[i].genes;
              children_[i].gradients = population[i].gradients;
            }
          }

          // genotype-phenotype mapping
          {
            BLOCKPROFILER("phenotype");
            // size_t gene_count = children_[0].genes.size();
            genotypes_.resize(child_count);
            for (size_t i = 0; i < child_count; i++)
              genotypes_[i] = children_[i].genes.data();
            model_.computeApproximateMutations(child_count, genotypes_.data(),
                                               phenotypes_);
          }

          // fitness
          {
            BLOCKPROFILER("fitness");
            for (size_t child_index = 0; child_index < child_count;
                 child_index++) {
              children_[child_index].fitness = computeFitnessActiveVariables(
                  phenotypes_[child_index], genotypes_[child_index]);
            }
          }

          // selection
          {
            BLOCKPROFILER("selection");
            child_indices_.resize(child_count);
            for (size_t i = 0; i < child_count; i++) child_indices_[i] = i;
            for (size_t i = 0; i < population.size(); i++) {
              size_t jmin = i;
              double fmin = children_[child_indices_[i]].fitness;
              for (size_t j = i + 1; j < child_count; j++) {
                double f = children_[child_indices_[j]].fitness;
                if (f < fmin) jmin = j, fmin = f;
              }
              std::swap(child_indices_[i], child_indices_[jmin]);
            }
            for (size_t i = 0; i < population.size(); i++) {
              std::swap(population[i].genes,
                        children_[child_indices_[i]].genes);
              std::swap(population[i].gradients,
                        children_[child_indices_[i]].gradients);
            }
          }
        }
      }

      // memetic optimization
      {
        BLOCKPROFILER("memetics");

        if (memetic == 'q' || memetic == 'l') {
          // init
          auto& individual = population[0];
          gradient_.resize(problem_.active_variables.size());
          if (genotypes_.empty()) genotypes_.emplace_back();
          phenotypes2_.resize(1);
          phenotypes3_.resize(1);

          // differentiation step size
          double dp = 0.0000001;
          if (fast_random() < 0.5) dp = -dp;

          for (size_t generation = 0; generation < 8; generation++)
          // for(size_t generation = 0; generation < 32; generation++)
          {
            if (canceled_) break;

            // compute gradient
            temp_ = individual.genes;
            genotypes_[0] = temp_.data();
            model_.computeApproximateMutations(1, genotypes_.data(),
                                               phenotypes2_);
            double f2p =
                computeFitnessActiveVariables(phenotypes2_[0], genotypes_[0]);
            double fa =
                f2p + computeSecondaryFitnessActiveVariables(genotypes_[0]);
            for (size_t i = 0; i < problem_.active_variables.size(); i++) {
              // double* pp = &(genotypes_[0][i]);
              genotypes_[0][i] = individual.genes[i] + dp;
              model_.computeApproximateMutation1(problem_.active_variables[i],
                                                 +dp, phenotypes2_[0],
                                                 phenotypes3_[0]);
              double fb = computeCombinedFitnessActiveVariables(phenotypes3_[0],
                                                                genotypes_[0]);
              genotypes_[0][i] = individual.genes[i];
              double d = fb - fa;
              gradient_[i] = d;
            }

            // normalize gradient
            const double sum = [&] {
              double result = dp * dp;
              for (size_t i = 0; i < problem_.active_variables.size(); i++)
                result += fabs(gradient_[i]);
              return result;
            }();
            {
              const double f = 1.0 / sum * dp;
              for (size_t i = 0; i < problem_.active_variables.size(); i++)
                gradient_[i] *= f;
            }

            // sample support points for line search
            for (size_t i = 0; i < problem_.active_variables.size(); i++)
              genotypes_[0][i] = individual.genes[i] - gradient_[i];
            model_.computeApproximateMutations(1, genotypes_.data(),
                                               phenotypes3_);
            double f1 = computeCombinedFitnessActiveVariables(phenotypes3_[0],
                                                              genotypes_[0]);

            double f2 = fa;

            for (size_t i = 0; i < problem_.active_variables.size(); i++)
              genotypes_[0][i] = individual.genes[i] + gradient_[i];
            model_.computeApproximateMutations(1, genotypes_.data(),
                                               phenotypes3_);
            double f3 = computeCombinedFitnessActiveVariables(phenotypes3_[0],
                                                              genotypes_[0]);

            // quadratic step size
            if (memetic == 'q') {
              // compute step size
              double v1 = (f2 - f1);       // f / j
              double v2 = (f3 - f2);       // f / j
              double v = (v1 + v2) * 0.5;  // f / j
              double a = (v1 - v2);        // f / j^2
              double step_size =
                  v / a;  // (f / j) / (f / j^2) = f / j / f * j * j = j

              // double v1 = (f2 - f1) / dp;
              // double v2 = (f3 - f2) / dp;
              // double v = (v1 + v2) * 0.5;
              // double a = (v2 - v1) / dp;
              // // v * x + a * x * x = 0;
              // // v = - a * x
              // // - v / a = x
              // // x = -v / a;
              // double step_size = -v / a / dp;

              // for(double f : { 1.0, 0.5, 0.25 })
              {
                double f = 1.0;

                // move by step size along gradient and compute fitness
                for (size_t i = 0; i < problem_.active_variables.size(); i++)
                  genotypes_[0][i] = modelInfo_.clip(
                      individual.genes[i] + gradient_[i] * step_size * f,
                      problem_.active_variables[i]);
                model_.computeApproximateMutations(1, genotypes_.data(),
                                                   phenotypes2_);
                double f4p = computeFitnessActiveVariables(phenotypes2_[0],
                                                           genotypes_[0]);

                // accept new position if better
                if (f4p < f2p) {
                  individual.genes = temp_;
                  continue;
                } else {
                  break;
                }
              }

              // break;
            }

            // linear step size
            if (memetic == 'l') {
              // compute step size
              double cost_diff = (f3 - f1) * 0.5;  // f / j
              double step_size = f2 / cost_diff;   // f / (f / j) = j

              // move by step size along gradient and compute fitness
              for (size_t i = 0; i < problem_.active_variables.size(); i++)
                temp_[i] = modelInfo_.clip(
                    individual.genes[i] - gradient_[i] * step_size,
                    problem_.active_variables[i]);
              model_.computeApproximateMutations(1, genotypes_.data(),
                                                 phenotypes2_);
              double f4p =
                  computeFitnessActiveVariables(phenotypes2_[0], genotypes_[0]);

              // accept new position if better
              if (f4p < f2p) {
                individual.genes = temp_;
                continue;
              } else {
                break;
              }
            }
          }
        }
      }
    }

    {
      BLOCKPROFILER("single_species");

      // compute single_species fitness
      for (auto& single_species : species_) {
        genesToJointVariables(single_species.individuals[0],
                              temp_joint_variables_);
        double fitness = computeFitness(temp_joint_variables_);
        single_species.improved = (fitness != single_species.fitness);
        single_species.fitness = fitness;
      }

      // sort single_species by fitness
      std::sort(species_.begin(), species_.end(),
                [](const Species& a, const Species& b) {
                  return a.fitness < b.fitness;
                });

      // wipeouts
      for (size_t species_index = 1; species_index < species_.size();
           species_index++) {
        if (fast_random() < 0.1 || !species_[species_index].improved)
        // if(fast_random() < 0.05 || !single_species[species_index].improved)
        {
          {
            auto& individual = species_[species_index].individuals[0];

            for (size_t i = 0; i < individual.genes.size(); i++)
              individual.genes[i] =
                  random(modelInfo_.getMin(problem_.active_variables[i]),
                         modelInfo_.getMax(problem_.active_variables[i]));

            for (auto& v : individual.gradients) v = 0;
          }
          for (size_t i = 0; i < species_[species_index].individuals.size();
               i++)
            species_[species_index].individuals[i] =
                species_[species_index].individuals[0];
        }
      }

      // update solution
      if (species_[0].fitness < solution_fitness_) {
        genesToJointVariables(species_[0].individuals[0], solution_);
        solution_fitness_ = species_[0].fitness;
      }
    }
  }

  // number of islands
  virtual size_t concurrency() const { return 4; }
};

std::optional<std::unique_ptr<IKSolver>> makeEvolution2Solver(
    const IKParams& params) {
  const auto& name = params.ros_params.mode;
  if (name == "bio2")
    return std::make_unique<IKEvolution2<0>>(params);
  else if (name == "bio2_memetic")
    return std::make_unique<IKEvolution2<'q'>>(params);
  else if (name == "bio2_memetic_l")
    return std::make_unique<IKEvolution2<'l'>>(params);
  else
    return std::nullopt;
}

}  // namespace bio_ik
