import 'package:dio/dio.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:yourstock/models/stocks.dart';

class StocksCubit extends Cubit<StocksState> {
  final String symbol;

  StocksCubit(this.symbol) : super(StocksInitial());
  final Dio _dio = Dio();

  Future<void> getDataminute({String? function}) async {
    emit(StocksLoading());
    try {
      final response = await _dio.get(
        "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=$symbol&interval=1min&apikey=0G0BN7WU6YYWLWMI",
      );
      if (response.statusCode == 200) {
        final stocks =
            Stocks.fromMap(response.data, function: "Time Series (1min)");
        final date = response.data["Time Series (1min)"];
        final chartData = convertToChartData(date);

        emit(StocksLoaded(
            stocks: stocks, date: date.keys.toList(), chartData: chartData));
      } else {
        emit(StocksError('Error: ${response.statusCode}'));
      }
    } catch (e) {
      emit(StocksError(
          'You reached the maximum number of requests, wait for 1 minute.'));
    }
  }

  Future<void> getDataWeekly({String? function}) async {
    emit(StocksLoading());
    try {
      final response = await _dio.get(
        "https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=$symbol&apikey=JFF5XUTNZIEADCG0",
      );
      if (response.statusCode == 200) {
        final stocks =
            Stocks.fromMap(response.data, function: "Weekly Time Series");
        final date = response.data["Weekly Time Series"];
        final chartData = convertToChartData(date);

        emit(StocksLoaded(
            stocks: stocks, date: date.keys.toList(), chartData: chartData));
      } else {
        emit(StocksError('Error: ${response.statusCode}'));
      }
    } catch (e) {
      emit(StocksError(
          'You reached the maximum number of requests, wait for 1 minute.'));
    }
  }

  Future<void> getDataMonthly({String? function}) async {
    emit(StocksLoading());
    try {
      final response = await _dio.get(
        "https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=$symbol&apikey=POCMQ0KHKRNQ7JO6",
      );
      if (response.statusCode == 200) {
        final stocks =
            Stocks.fromMap(response.data, function: "Monthly Time Series");
        final date = response.data["Monthly Time Series"];
        final chartData = convertToChartData(date);

        emit(StocksLoaded(
            stocks: stocks, date: date.keys.toList(), chartData: chartData));
      } else {
        emit(StocksError('Error: ${response.statusCode}'));
      }
    } catch (e) {
      emit(StocksError(
          'You reached the maximum number of requests, wait for 1 minute.'));
    }
  }

  Future<void> getDescriptionData() async {
    emit(StocksLoading());
    try {
      final response = await _dio.get(
        "https://www.alphavantage.co/query?function=OVERVIEW&symbol=${symbol}&apikey=8TNLUH11I1ZYP5EW",
      );
      if (response.statusCode == 200) {
        final data = response.data["Description"];

        emit(DescriptionLoaded(description: data));
      } else {
        emit(StocksError('Error: ${response.statusCode}'));
      }
    } catch (e) {
      emit(StocksError(
          'You reached the maximum number of requests, wait for 1 minute.'));
    }
  }
}

List<DataPoint> convertToChartData(Map<String, dynamic> date) {
  final List<DataPoint> chartData = [];
  date.forEach((key, value) {
    final DateTime xValue = DateTime.parse(key);
    final double open = double.parse(value['1. open']);
    final double close = double.parse(value['4. close']);
    final double high = double.parse(value['2. high']);
    final double low = double.parse(value['3. low']);
    chartData.add(DataPoint(xValue, open, close, high, low));
  });
  return chartData;
}

abstract class StocksState {}

class DescriptionLoading extends StocksState {}

class DescriptionLoaded extends StocksState {
  final String description;

  DescriptionLoaded({required this.description});
}

class DescriptionError extends StocksState {
  final String error;

  DescriptionError(this.error);
}

class StocksInitial extends StocksState {}

class StocksLoading extends StocksState {}

class StocksLoaded extends StocksState {
  final Stocks stocks;
  final List<String> date;
  final List<DataPoint> chartData;

  StocksLoaded(
      {required this.stocks, required this.date, required this.chartData});
}

class StocksError extends StocksState {
  final String error;

  StocksError(this.error);
}

class DataPoint {
  final DateTime xValue;
  final double open;
  final double close;
  final double high;
  final double low;

  DataPoint(this.xValue, this.open, this.close, this.high, this.low);
}
