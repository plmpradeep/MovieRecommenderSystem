var app=angular.module('mrs', ['ngAnimate', 'ngSanitize', 'ui.bootstrap']);
app.controller('TypeaheadCtrl', function($scope, $http) {

  $scope.movies;
  $scope.selectedMovie = undefined;

$scope.getMovies = function() {
    $http({
        method: 'GET',
        url: '/movies',

    }).then(function(response) {
        $scope.movies = response.data;
    }, function(error) {
        console.log(error);
    });
}

$scope.getTFMovies = function(movie) {
  $http({
      method: 'POST',
      url: '/tfidf',
      data: {
        movie: movie
            }

  }).then(function(response) {
      $scope.tfMovies = response.data;
  }, function(error) {
      console.log(error);
  });
}

$scope.getPearsonMovies = function(movie) {
  $http({
      method: 'POST',
      url: '/pearson',
      data: {
        movie: movie
            }

  }).then(function(response) {
      $scope.pearsonMovies = response.data;
  }, function(error) {
      console.log(error);
  });
}


$scope.getSVDMovies = function() {
  $http({
      method: 'POST',
      url: '/svdByUserID',
      data: {
        userID: $scope.userId
            }

  }).then(function(response) {
      $scope.svdMovies = response.data;
  }, function(error) {
      console.log(error);
  });
}


$scope.getMovies();

$scope.onSelect = function ($item) {
 $scope.selectedMovie = $item;
 $scope.getTFMovies($scope.selectedMovie);
 $scope.getPearsonMovies($scope.selectedMovie);
};

});
