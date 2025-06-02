import 'package:yourstock/data/models/search_modeel.dart';

abstract class SearchState {}

class SearchInitial extends SearchState {}

class SearchLoaded extends SearchState {
  final SearchSymbol searchSymbol;

  SearchLoaded(this.searchSymbol);
}

class SearchLoading extends SearchState {}

class SearchError extends SearchState {}
