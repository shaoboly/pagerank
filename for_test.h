#ifndef for_test_h
#define for_test_h

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class vertex
{
public:
	int id;
	int in_deg;
	int out_deg;
	vector<int> in_edge;
	vector<int> out_edge;
};

void bin_uncconverge_count_test(float *all_diff, vector<vector<int>> all_bin, int bins_number, int iteration_number, ofstream &out)
{
	
	int unconverge_number = 0;
	int bin_uconverge;

	if (iteration_number == 0)
	{
		out << "bin_number,";
		for (int i = 0; i < bins_number; i++)
		{
			out << all_bin[i].size() << ",";
		}
		out << endl;
	}

	out << iteration_number << ",";
	for (int i = 0; i < bins_number; i++)
	{
		bin_uconverge = 0;
		for (int j = 0; j < all_bin[i].size(); j++)
		{
			int node = all_bin[i][j];
			if (abs(all_diff[node]) > 0.00001)
				bin_uconverge += 1;
		}
		out << (float)bin_uconverge / all_bin[i].size() << ",";
	}
	out << endl;
}

void bin_uncconverge_precision(float *all_diff, vector<vector<int>> all_bin, int bins_number, int iteration_number, ofstream &out)
{

	int unconverge_number = 0;
	float bin_diff;
	float sum_diff;
	int sum_num = 0;

	if (iteration_number == 0)
	{
		out << "bin_number,";
		for (int i = 0; i < bins_number; i++)
		{
			out << all_bin[i].size() << ",";
		}
		out << endl;
	}

	out << iteration_number << ",";

	sum_diff = 0;
	for (int i = 0; i < bins_number; i++)
	{
		bin_diff = 0;
		sum_num += all_bin[i].size();
		for (int j = 0; j < all_bin[i].size(); j++)
		{
			int node = all_bin[i][j];
			bin_diff += abs(all_diff[node]);
			sum_diff += abs(all_diff[node]);
		}
		out << (float)bin_diff / all_bin[i].size() << ",";
	}
	out << sum_diff /sum_num<< endl;
}

void deal_with_g1_batch(vector<vertex> &vertices, vector<int> &host_g1,int row_max,vector<int> &batch_label,int workload,int &batch_block_number )
{
	cout << "host_g1" << host_g1.size() << endl;
	cout << "row_max" << row_max << endl;
	int tmp_sum = 0;
	
	for (int i = 0; i < host_g1.size(); i++)
	{
		if (tmp_sum==0)
			batch_label.push_back(i);
		tmp_sum += vertices[host_g1[i]].in_deg;
		if (tmp_sum >= workload)
		{
			tmp_sum = 0;
		}
	}
	batch_label.push_back(host_g1.size());

	cout << "batch_label" << batch_label.size() << endl;

	batch_block_number = (batch_label.size()-1 + 1023) / 1024;
}

#endif