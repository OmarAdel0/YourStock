import 'package:dio/dio.dart';

class NewsWebServices {
  late Dio dio;

  NewsWebServices() {
    BaseOptions options = BaseOptions(
      baseUrl: 'https://www.alphavantage.co/',
      receiveDataWhenStatusError: true,
    );
    dio = Dio(options);
  }

  Future<List<dynamic>> getAllNews() async {
    try {
      Response response = await dio.get(
          'query?function=NEWS_SENTIMENT&keywords=&language=en&apikey=0G0BN7WU6YYWLWMI');
      return response.data['feed'];
    } catch (e) {
      return [];
    }
  }
}