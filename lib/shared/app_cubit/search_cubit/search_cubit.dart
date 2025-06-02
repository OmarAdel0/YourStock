import 'dart:async';
import 'package:bloc/bloc.dart';
import 'package:yourstock/data/models/search_modeel.dart';
import 'package:yourstock/shared/app_cubit/search_cubit/search_state.dart';
import 'package:dio/dio.dart';
import 'package:rxdart/rxdart.dart';

class SearchCubit extends Cubit<SearchState> {
  final Dio dio = Dio();

  late final BehaviorSubject<String> _searchQueryController;
  StreamSubscription<String>? _searchQuerySubscription;

  SearchCubit() : super(SearchInitial()) {
    _searchQueryController = BehaviorSubject<String>();
    _searchQuerySubscription = _searchQueryController
        .debounceTime(const Duration(milliseconds: 300))
        .distinct()
        .listen((symbol) {
      searchData(symbol: symbol);
    });
  }

  Future<void> searchData({String? symbol}) async {
    emit(SearchLoading());

    String url =
        "https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=$symbol&apikey=0G0BN7WU6YYWLWMI";

    try {
      Response response = await dio.get(url);
      SearchSymbol searchSymbol = SearchSymbol.fromMap(response.data);
      emit(SearchLoaded(searchSymbol));
    } catch (e) {
      emit(SearchError());
    }
  }

  void setSearchQuery(String query) {
    _searchQueryController.add(query);
  }

  @override
  Future<void> close() {
    _searchQuerySubscription?.cancel();
    _searchQueryController.close();
    return super.close();
  }
}
