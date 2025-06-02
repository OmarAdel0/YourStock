// ignore_for_file: public_member_api_docs, sort_constructors_first
import 'package:bloc/bloc.dart';
import 'package:yourstock/data/repository/news_stock_repository.dart';
import 'package:yourstock/shared/app_cubit/news_cubit/news_cubit.dart';
import 'package:yourstock/data/models/news.dart';

class NewsStockCubit extends Cubit<NewsState> {
  final NewsStockRepository newsStockRepository;
  final String symbol;
  List<News> news = [];

  NewsStockCubit(this.newsStockRepository, this.symbol) : super(NewsInitial());

  List<News> getAllNews({String? symbol}) {
    emit(NewsLoading());
    newsStockRepository.getAllNews(symbol: symbol).then((news) {
      emit(NewsLoaded(news));
      this.news = news;
    });
    return news;
  }
}
