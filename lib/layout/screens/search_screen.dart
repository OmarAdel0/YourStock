import 'package:yourstock/layout/screens/chart_screen.dart';
import 'package:yourstock/shared/app_cubit/search_cubit/search_cubit.dart';
import 'package:yourstock/shared/app_cubit/search_cubit/search_state.dart';
import 'package:yourstock/shared/componentes/components.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class SearchScreen extends StatelessWidget {
  final SearchCubit searchCubit = SearchCubit();

  SearchScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<SearchCubit, SearchState>(
      bloc: searchCubit,
      builder: (BuildContext context, SearchState state) {
        return Scaffold(
          appBar: AppBar(
            title: const Text(""),
          ),
          body: Padding(
            padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 15),
            child: Column(
              children: [
                TextFormField(
                  keyboardType: TextInputType.text,
                  decoration: InputDecoration(
                    labelText: "Search",
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                    prefixIcon: const Icon(Icons.search),
                  ),
                  onChanged: (value) {
                    searchCubit.setSearchQuery(value.isNotEmpty
                        ? value
                        : "Couldn't find what you are searching for");
                  },
                ),
                const SizedBox(height: 8),
                Expanded(
                  child: _buildSearchResults(state),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildSearchResults(SearchState state) {
    if (state is SearchLoading) {
      return const Center(child: CircularProgressIndicator());
    } else if (state is SearchLoaded) {
      return ListView.builder(
        itemCount: state.searchSymbol.bestMatches?.length ?? 0,
        itemBuilder: (BuildContext context, int index) {
          var match = state.searchSymbol.bestMatches![index];
          if (match.region == "United States") {
            return MaterialButton(
              onPressed: () {
                navigateTo(
                    context,
                    ChartScreen(
                        symbol: '${match.symbol}'));
              },
              child: ListTile(
                title: Text(
                  match.name ?? "No Name",
                ),
                subtitle: Text(
                  match.symbol ?? "No Symbol",
                ),
                trailing: Text(
                  match.region ?? "No Region",
                ),
              ),
            );
          } else {
            return const SizedBox.shrink();
          }
        },
      );
    } else {
      return const SizedBox.shrink();
    }
  }
}
