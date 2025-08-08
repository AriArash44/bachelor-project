import useSWR from 'swr';
import { axiosFetcher } from '../utils/fetcher';

export function useFetcher(endpoint, options = {}) {
  return useSWR(endpoint, axiosFetcher, {
    refreshInterval: 10000,
    ...options,
  });
}
