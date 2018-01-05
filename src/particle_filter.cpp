/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    //Initialize the number of particles
	num_particles = 100;
	
	// create normal distributions for x, y and theta
  	std::normal_distribution<double> dist_x(x, std[0]);
  	std::normal_distribution<double> dist_y(y, std[1]);
  	std::normal_distribution<double> dist_theta(theta, std[2]);
    
    
    // Generate particles with normal distribution with mean on GPS values.
    for (int i = 0; i < num_particles; i++) {
    	Particle particle;
    	particle.id = i;
   		particle.x = dist_x(gen);
   		particle.y = dist_y(gen);
    	particle.theta = dist_theta(gen);
    	particle.weight = 1.0;
		particles.push_back(particle);
	}

  	// Finish initialization 
  	is_initialized = true;
}



void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Predicate each particle x, y and theta 
	double x, y, theta; 
    for (int i = 0; i < num_particles; i++) {
    	
    	theta = particles[i].theta;
    	//yaw_rate = 0
       	if ( fabs(yaw_rate) < .0001 ) { 
      		x = particles[i].x + velocity * delta_t * cos( theta );
      		y = particles[i].y += velocity * delta_t * sin( theta );
     		//particle[i].theta remains the same
      	} 
      	else { //yaw_rate ~= 0
      		x = particles[i].x + velocity / yaw_rate * ( sin( theta + yaw_rate * delta_t ) - sin( theta ) );
      		y = particles[i].y + velocity / yaw_rate * ( cos( theta ) - cos( theta + yaw_rate * delta_t ) );
      		theta = particles[i].theta + yaw_rate * delta_t;
    	}

    // Create normal distributions for x, y and theta
  	std::normal_distribution<double> dist_x(x, std_pos[0]);
  	std::normal_distribution<double> dist_y(y, std_pos[1]);
  	std::normal_distribution<double> dist_theta(theta, std_pos[2]);
    
    // Add Random Guassian noise.
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}



void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	//Associate Measurements to Landmarks using Nearest Neighbor
	int num_predicted = predicted.size();
	int num_observations = observations.size();
	int obs_id = -1;
	for (int i = 0; i < num_observations; i++) {
		double minDistance = std::numeric_limits<double>::max(); 
		for (int j = 0; j < num_predicted; j++) {
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (distance < minDistance) {
				minDistance = distance;
				obs_id = predicted[j].id;
			}
		}
		observations[i].id = obs_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  for(auto& particle: particles){
    double x = particle.x;
    double y = particle.y;
    double theta = particle.theta;

    // Find the landmarks in the particle's range 
    vector<LandmarkObs> landmarksInRange;
    for(const auto& landmark: map_landmarks.landmark_list){
      double x_f = landmark.x_f;
      double y_f = landmark.y_f;
      int id = landmark.id_i;
      double distance = dist(x, y, x_f, y_f);
      if( distance < sensor_range){ 
        landmarksInRange.push_back(LandmarkObs{id, x_f, y_f});
      }
    }

    //Tranform the observations from vehicle's system into the map system 
    vector<LandmarkObs> observationsInMapSystem;
    for(const auto& obs: observations){
      double x_map = x + obs.x * cos(theta) - obs.y * sin(theta);
      double y_map = y + obs.x * sin(theta) + obs.y * cos(theta);
      observationsInMapSystem.push_back(LandmarkObs{obs.id, x_map, y_map});
    }

    // Associate Observation to the NN landmark
    dataAssociation(landmarksInRange, observationsInMapSystem);

    //Update the weight of each particle using a mult-variate Gaussian distribution
    particle.weight = 1.0;
    for(const auto& obs_m: observationsInMapSystem){
      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);    
      double exponent =  pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2)) + pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      double prob = exp(-exponent) / (2 * M_PI * std_landmark[0] * std_landmark[1]);      
      particle.weight *=  prob;
    }
    weights.push_back(particle.weight);
  }
}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	//resample : refer from the above link
	//std::discrete_distribution produces random integers on the interval [0, n), where the probability of each individual integer i is defined as w
    //i/S, that is the weight of the ith integer divided by the sum of all n weights.
	vector<Particle> resampledParticles;
	resampledParticles.resize(num_particles);
	std::random_device rd;
	std::mt19937 gen(rd());
    std::discrete_distribution<> dist(weights.begin(), weights.end());
    for(int i = 0; i < num_particles; i++){
    	int index = dist(gen);
    	resampledParticles[i] = particles[index];
  	}
  	particles = resampledParticles;
  	weights.clear();
 }


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
