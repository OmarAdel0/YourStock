import 'package:dio/dio.dart';

class DescriptionWebServices {
  late Dio dio;

  DescriptionWebServices() {
    BaseOptions options = BaseOptions(
      baseUrl: 'https://www.alphavantage.co/',
      receiveDataWhenStatusError: true,
    );
    dio = Dio(options);
  }

  Future<List<dynamic>> getDescribtion() async {
    try {
      Response response = await dio
          .get('query?function=OVERVIEW&symbol=AMZN&apikey=0G0BN7WU6YYWLWMI');
      return response.data['Description'];
    } catch (e) {
      return [];
    }
  }
}
