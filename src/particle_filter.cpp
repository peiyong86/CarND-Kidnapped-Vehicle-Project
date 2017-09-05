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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

	for(int i=0; i<num_particles; i++){
		double sample_x, sample_y, sample_theta;
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		Particle p;
		p.id = i;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;

		particles.push_back(p);
		weights.push_back(1.0);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	if(abs(yaw_rate) > 0.001){
		for(int i=0; i<num_particles; i++){
			double theta = particles[i].theta;
			particles[i].x += velocity/yaw_rate*(sin(theta+delta_t*yaw_rate) - sin(theta)) + dist_x(gen);
			particles[i].y += velocity/yaw_rate*(cos(theta) - cos(theta+delta_t*yaw_rate)) + dist_y(gen);
			particles[i].theta += delta_t*yaw_rate + dist_theta(gen);
		}
	}
	else{
		for(int i=0; i<num_particles; i++){
			double theta = particles[i].theta;
			particles[i].x += velocity*delta_t*cos(theta) + dist_x(gen);
			particles[i].y += velocity*delta_t*sin(theta) + dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i = 0; i<observations.size(); i++){
		int index = 0;
		double mindis = 99999999.0;
		double dist_ = 0;
		for(int j = 0; j<predicted.size(); j++){
			dist_ = dist(observations[i].x, observations[i].y, predicted[i].x, predicted[i].y);
			if(dist_ < mindis){
				mindis = dist_;
				index = j;
			}
		}
		observations[i].id = index;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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
	for(int i = 0; i < num_particles; i++) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		vector<LandmarkObs> observations_copy(observations);
		double ox, oy;
		for (int j = 0; j < observations_copy.size(); j++) {
			double ox = cos(theta) * observations_copy[j].x - sin(theta) * observations_copy[j].y + x;
			double oy = sin(theta) * observations_copy[j].x + cos(theta) * observations_copy[j].y + y;
			observations_copy[j].x = ox;
			observations_copy[j].y = oy;
		}

		vector<LandmarkObs> landmarks;
        vector<double> distances;
		double dist_;
		int index = 0;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			dist_ = dist(x, y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
			if (dist_ < sensor_range) {
				landmarks.push_back(
						LandmarkObs{index, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
				distances.push_back(dist_);
				index++;
			}
		}
		dataAssociation(observations_copy, landmarks);

		double prob = 1.0;
		double exponent;
		double gaussian = 0;
		double gauss_norm = 1.0 / sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
        for(int j = 0; j < landmarks.size(); j++){
			exponent = pow(observations_copy[landmarks[j].id].x - landmarks[j].x, 2)/(2*pow(std_landmark[0], 2));
			exponent += pow(observations_copy[landmarks[j].id].y - landmarks[j].y, 2)/(2*pow(std_landmark[1], 2));
			gaussian = exp(-exponent)*gauss_norm;
			prob *= gaussian;
        }
		weights[i] = prob;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    double w_sum = 0, w_max=weights[0];
    for(int i = 0; i < weights.size(); i++){
        w_sum += weights[i];
		if(weights[i]>w_max){
			w_max = weights[i];
		}
    }
	for(int i = 0; i < weights.size(); i++) {
		weights[i] /= w_sum;
	}
	w_max /= w_sum;

	default_random_engine gen;
	uniform_real_distribution<double> randomdistri(0.0, 2*w_max);

	int index = 0;
	double step;
	vector<Particle> new_particles;
	step = randomdistri(gen);
	while(new_particles.size() < num_particles){
		if(step>weights[index]){
			step = step - weights[index];
			index += 1;
			index = index % num_particles;
		}
		else{
			new_particles.push_back(particles[index]);
			step += randomdistri(gen);
		}
	}
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
