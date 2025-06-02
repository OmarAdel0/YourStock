import 'package:dio/dio.dart';

class DioHelper {
  static Dio? dio;

  static init() {
    dio = Dio();
  }

  static Future<Response?> get({required String url}) async
  {
    return await dio?.get(url);
  }

}


//   try{
//   final response = await dio?.get(url);
//   return response?.data;
//   }catch(error, stacktrace){
//   log('Exception occured: $error with stacktrace: $stacktrace');
//   throw Exception('Exception occured: $error with stacktrace: $stacktrace');
//   }
// }