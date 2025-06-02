import 'package:yourstock/models/search_model.dart';

abstract class AppStates {}

class AppInitialState extends AppStates {}

class AppBottomNavState extends AppStates {}

class GetStocksLoadingState extends AppStates {}

class StocksGetStocksSuccessState extends AppStates {}

class StocksGetStocksErrorState extends AppStates {
  late final String error;
  StocksGetStocksErrorState(this.error);
}

class StocksGetStockSymbolState extends AppStates {}

class SearchState extends AppStates {
  final SearchSymbol? searchSymbol;

  SearchState({this.searchSymbol});
}

class AppLoadingState extends AppStates {}
