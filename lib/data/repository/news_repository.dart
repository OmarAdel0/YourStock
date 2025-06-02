import 'package:yourstock/data/models/news.dart';
import 'package:yourstock/data/web_services/news_web_services.dart';

class NewsRepository {
  final NewsWebServices newsWebServices;

  NewsRepository(this.newsWebServices);

  Future<List<News>> getAllNews() async {
    final news = await newsWebServices.getAllNews();
    return news.map((news) => News.fromJson(news)).toList();
  }
}
