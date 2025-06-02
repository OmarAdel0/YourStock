import 'package:dio/dio.dart';

class NewsStockWebServices {
  late Dio dio;

  NewsStockWebServices() {
    BaseOptions options = BaseOptions(
      baseUrl: 'https://www.alphavantage.co/',
      receiveDataWhenStatusError: true,
    );
    dio = Dio(options);
  }

  Future<List<dynamic>> getAllNews({String? symbol}) async {
    try {
      Response response = await dio.get(
          'query?function=NEWS_SENTIMENT&tickers=$symbol&time_from=20220410T0130&limit=200&apikey=0OCDJ0Z92R3UK7VE');
      //print(response.data['feed'].toString());
      return response.data['feed'];
    } catch (e) {
      return [];
    }
  }
}