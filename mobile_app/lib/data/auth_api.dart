import 'package:dio/dio.dart';

import '../core/api_config.dart';

class AuthResult {
  const AuthResult({
    required this.success,
    required this.message,
  });

  final bool success;
  final String message;
}

class AuthApi {
  AuthApi({Dio? dio})
      : _dio = dio ??
            Dio(
              BaseOptions(
                baseUrl: ApiConfig.baseUrl,
                connectTimeout: const Duration(seconds: 10),
                receiveTimeout: const Duration(seconds: 15),
                sendTimeout: const Duration(seconds: 15),
                headers: const {
                  'Content-Type': 'application/json',
                  'Accept': 'application/json',
                },
              ),
            );

  final Dio _dio;

  Future<AuthResult> login({
    required String email,
    required String password,
  }) async {
    try {
      final response = await _dio.post<dynamic>(
        '/auth/login',
        data: {
          'email': email.trim(),
          'password': password,
        },
      );

      final data = response.data;
      if (data is Map<String, dynamic>) {
        final hasAccessToken = data['accessToken'] != null || data['access_token'] != null;
        if (hasAccessToken) {
          return const AuthResult(success: true, message: 'Вход выполнен успешно');
        }
      }

      return const AuthResult(
        success: true,
        message: 'Ответ сервера получен. Подключим токены следующим шагом.',
      );
    } on DioException catch (e) {
      final statusCode = e.response?.statusCode;
      final responseData = e.response?.data;

      if (statusCode == 401 || statusCode == 400) {
        return const AuthResult(success: false, message: 'Неверный email или пароль');
      }

      if (responseData is Map<String, dynamic>) {
        final message = responseData['message']?.toString();
        if (message != null && message.isNotEmpty) {
          return AuthResult(success: false, message: message);
        }
      }

      return AuthResult(
        success: false,
        message: 'Ошибка сети: ${e.message ?? 'попробуйте позже'}',
      );
    } catch (_) {
      return const AuthResult(success: false, message: 'Неизвестная ошибка входа');
    }
  }
}
