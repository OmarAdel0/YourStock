part of 'news_cubit.dart';

@immutable
abstract class NewsState {}

class NewsInitial extends NewsState {}

class NewsLoaded extends NewsState {
  final List<News> news;

  NewsLoaded(this.news);
}

class NewsSuccessState extends NewsState{}
class NewsLoading extends NewsState{}
class NewsError extends NewsState{}